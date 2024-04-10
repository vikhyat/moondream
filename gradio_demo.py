import argparse
import torch
import re
from moondream import detect_device, LATEST_REVISION
import gradio as gr
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
parser.add_argument("--model", type=str, help="Path to the model directory")
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

if args.model:
    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    last_folder = args.model.split('/')[-1]
    if "moondream" in last_folder:
        moondream = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device=device, dtype=dtype)
    else:
        moondream = AutoModelForCausalLM.from_pretrained(model_id).to(device=device, dtype=dtype)
else:
    model_id = "vikhyatk/moondream2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
    moondream = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=LATEST_REVISION).to(device=device, dtype=dtype)
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
        clean_text = re.sub("<$|<END$", "", new_text)
        buffer += clean_text
        yield buffer


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌔 moondream
        ### A tiny vision language model. GitHub
        """
    )
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        with gr.Column():
            output = gr.TextArea(label="Response")
            prompt = gr.Textbox(label="Input Prompt", placeholder="Type here...")
            submit = gr.Button("Submit")

    submit.click(answer_question, [img, prompt], output)
    prompt.submit(answer_question, [img, prompt], output)

demo.queue().launch(debug=True)
