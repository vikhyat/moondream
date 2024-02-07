import argparse
import torch
import re
import gradio as gr
from moondream import Moondream, detect_device
from threading import Thread
from transformers import TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer

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

model_id = "vikhyatk/moondream1"
tokenizer = Tokenizer.from_pretrained(model_id)
moondream = Moondream.from_pretrained(model_id).to(device=device, dtype=dtype)
moondream.eval()


def answer_question(img, prompt):
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
        prompt = gr.Textbox(label="Input Prompt", placeholder="Type here...", scale=4)
        submit = gr.Button("Submit")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        output = gr.TextArea(label="Response")
    submit.click(answer_question, [img, prompt], output)
    prompt.submit(answer_question, [img, prompt], output)

demo.queue().launch(debug=True)
