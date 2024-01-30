import torch
import re
import gradio as gr
from moondream import VisionEncoder, TextModel, detect_device
from huggingface_hub import snapshot_download
from threading import Thread
from transformers import TextIteratorStreamer

device, dtype = detect_device()
if device != torch.device("cpu"):
    print("Using device:", device)
    print("If you run into issues, pass the `--cpu` flag to this script.")
    print()

model_path = snapshot_download("vikhyatk/moondream1")
vision_encoder = VisionEncoder(model_path).to(device=device, dtype=dtype)
text_model = TextModel(model_path).to(device=device, dtype=dtype)


def moondream(img, prompt, max_tokens):
    image_embeds = vision_encoder(img)
    streamer = TextIteratorStreamer(text_model.tokenizer, skip_special_tokens=True)
    thread = Thread(
        target=text_model.answer_question,
        kwargs={
            "image_embeds": image_embeds, "question": prompt,
            "streamer": streamer, "max_new_tokens": max_tokens
        },
    )
    thread.start()

    buffer = ""
    for new_text in streamer:
        clean_text = re.sub("<$|END$", "", new_text)
        buffer += clean_text
        yield buffer.strip("<END")


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌔 moondream
        ### A tiny vision language model. [GitHub](https://github.com/vikhyat/moondream)
        """
    )
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Input Prompt", placeholder="Type here...")
            max_tokens = gr.Slider(label="Max tokens", minimum=128,
                                   maximum=2048, value=128)
        submit = gr.Button("Submit")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        output = gr.TextArea(label="Response", info="Please wait for a few seconds..")
    submit.click(moondream, [img, prompt, max_tokens], output)
    prompt.submit(moondream, [img, prompt, max_tokens], output)

demo.queue().launch(debug=True)
