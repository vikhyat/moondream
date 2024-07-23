import argparse
import torch
import gradio as gr
from moondream import detect_device, LATEST_REVISION
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from PIL import Image, ImageDraw
import re
from torchvision.transforms.v2 import Resize

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

if args.cpu:
    device = torch.device("cpu")
    dtype = torch.float32
else:
    device, dtype = detect_device()
    if device != torch.device("cpu"):
        print("Using device:", device)
        print("If you run into issues, pass the `--cpu` flag to this script.")
        print()

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION, torch_dtype=dtype
).to(device=device)
moondream.eval()


def answer_question(img, prompts):
    buffer = []
    for prompt in prompts:
        image_embeds = moondream.encode_image(img)
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        thread = Thread(
            target=moondream.answer_question,
            kwargs={
                "image_embeds": image_embeds,
                "question": prompt,
                "tokenizer": tokenizer,
                "streamer": streamer,
            },
        )
        thread.start()

        response = ""
        for new_text in streamer:
            response += new_text

        buffer.append(response)

    return buffer


def extract_floats(text):
    pattern = r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]"
    matches = re.findall(pattern, text)
    return [[float(num) for num in match] for match in matches]


def extract_bboxes(text):
    float_lists = extract_floats(text)
    return [tuple(float_list) for float_list in float_lists]


def process_answers(img, answers):
    bboxes = []
    for answer in answers:
        bboxes.extend(extract_bboxes(answer))
    if bboxes:
        draw_image = img.copy()
        draw_image.thumbnail((768, 768), Image.LANCZOS)
        width, height = draw_image.size
        draw = ImageDraw.Draw(draw_image)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1 * width), int(x2 * width)
            y1, y2 = int(y1 * height), int(y2 * height)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

        return gr.update(visible=True, value=draw_image)
    return gr.update(visible=False, value=None)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌔 moondream
        """
    )
    with gr.Row():
        prompt = gr.Textbox(label="Input Prompts (comma-separated)", placeholder="Type here...", scale=4)
        submit = gr.Button("Submit")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        with gr.Column():
            output = gr.Markdown(label="Response")
            ann = gr.Image(visible=False, label="Annotated Image")

    import re


    def on_submit(img, prompt_text):
        prompts = []
        match = re.search(r'(.*?)(bounding box:\s*)(.*)', prompt_text, re.IGNORECASE)
        if match:
            items_str = match.group(3).strip()
            items = items_str.split(',')
            prompts.extend([f"bounding box: {item.strip()}" for item in items])
        else:
            prompts.append(prompt_text)

        answers = answer_question(img, prompts)
        processed_image = process_answers(img, answers)
        return processed_image, ''.join(answers)


    submit.click(on_submit, [img, prompt], [ann, output])
    prompt.submit(on_submit, [img, prompt], [ann, output])

demo.queue().launch(debug=True)
