import time
from matplotlib.pyplot import hist
import uvicorn
import argparse

import torch
from transformers import TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer
from sse_starlette.sse import EventSourceResponse

from loguru import logger
from typing import List, Literal, Union, Tuple, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

import requests
import base64
from PIL import Image
from io import BytesIO
import re
from threading import Thread

from moondream import Moondream, detect_device
from contextlib import asynccontextmanager

# 请求
class TextContent(BaseModel):
    type: Literal["text"]
    text: str
class ImageUrl(BaseModel):
    url: str
class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl
ContentItem = Union[TextContent, ImageUrlContent]
class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Additional parameters
    repetition_penalty: Optional[float] = 1.0

# 响应
class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse
class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None

# 图片输入处理
def process_img(input_data):
    if isinstance(input_data, str):
        # URL
        if input_data.startswith("http://") or input_data.startswith("https://"):
            response = requests.get(input_data)
            image_data = response.content
            pil_image = Image.open(BytesIO(image_data)).convert('RGB')
        # base64
        elif input_data.startswith("data:image/"):
            base64_data = input_data.split(",")[1]
            image_data = base64.b64decode(base64_data)
            pil_image = Image.open(BytesIO(image_data)).convert('RGB')
        # img_path
        else:
            pil_image = Image.open(input_data)
    # PIL
    elif isinstance(input_data, Image.Image):
        pil_image = input_data
    else:
        raise ValueError("data type error")

    return pil_image

# 历史消息处理
def process_history_and_images(messages: List[ChatMessageInput]) -> Tuple[
    Optional[str], Optional[str], Optional[List[Image.Image]]]:

    def chat_history_to_prompt(history):
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += f"Question: {old_query}\n\nAnswer: {response}\n\n"
        return prompt

    last_user_texts = ''
    formatted_history = []
    image_list = []

    for i, message in enumerate(messages):
        role = message.role
        content = message.content

        if isinstance(content, list):  # text
            text_content = ' '.join(item.text for item in content if isinstance(item, TextContent))
        else:
            text_content = content

        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    image = process_img(image_url)
                    image_list.append(image)

        if role == 'user':
            if i == len(messages) - 1:  # last message
                last_user_texts = text_content
            else:
                formatted_history.append((text_content, ''))
        elif role == 'assistant':
            if formatted_history:
                if formatted_history[-1][1] != '':
                    assert False, f"the last texts is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, f"assistant reply before user"
        else:
            assert False, f"unrecognized role: {role}"

    history = chat_history_to_prompt(formatted_history)

    return last_user_texts, history, image_list


@torch.inference_mode()
# Moondrean推理
def generate_stream_moondream(params: dict):
    global model, tokenizer

    # 输入处理
    messages = params["messages"]

    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 128))

    prompt, history, image_list = process_history_and_images(messages)
    # 只处理最后一张图
    img = image_list[-1]

    # 构建输入
    '''
    answer_question(
            image_embeds,
            question,
            tokenizer,
            max_new_tokens,
            chat_history="",
            result_queue=None,
            **kwargs,
        )
    '''
    image_embeds = model.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kwargs = {
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "max_new_tokens": max_new_tokens,
            "chat_history":history,
            "repetition_penalty": repetition_penalty,
            "do_sample": False,
            "top_p": top_p,
            "streamer": streamer,
        }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    thread = Thread(
        target=model.answer_question,
        kwargs=gen_kwargs,
    )
    
    input_echo_len = 0
    total_len = 0
    # 启动推理
    thread.start()
    buffer = ""
    for new_text in streamer:
        clean_text = re.sub("<$|END$", "", new_text)
        buffer += clean_text
        yield {
            "text": buffer.strip("<END"),
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
                },
            }
    generated_ret ={
        "text": buffer.strip("<END"),
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
            },
        }
    yield generated_ret

# 单次响应
def generate_moondream(params: dict):
    for response in generate_stream_moondream(params):
        pass
    return response

# 流式响应
async def predict(model_id: str, params: dict):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_moondream(params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode
        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 生命周期管理器，结束清显存
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
app = FastAPI(lifespan=lifespan)
# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 对话路由
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    # 检查请求
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 128,
        echo=False,
        stream=request.stream,
    )

    # 流式响应
    if request.stream:
        generate = predict(request.model, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")

    # 单次响应
    response = generate_moondream(gen_params)
    usage = UsageInfo()
    message = ChatMessageResponse(
        role="assistant",
        content=response["text"],
    )
    logger.debug(f"==== message ====\n{message}")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=usage)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="moon")
args = parser.parse_args()
mod = args.model

if mod == "moon":
    MODEL_PATH = "vikhyatk/moondream1"

tokenizer = Tokenizer.from_pretrained(MODEL_PATH)

device, dtype = detect_device()
if device != torch.device("cpu"):
    print("Using device:", device)
    print("If you run into issues, pass the `--cpu` flag to this script.")

def load_mod(model_input):
    global model
    model = Moondream.from_pretrained(model_input).to(device=device, dtype=dtype).eval()

if __name__ == "__main__":

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

    load_mod(MODEL_PATH)

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
