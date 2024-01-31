import torch
import argparse
from PIL import Image
from moondream import Moondream, detect_device
from queue import Queue
from threading import Thread
from transformers import (
    TextIteratorStreamer,
    CodeGenTokenizerFast as Tokenizer,
)
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=False)
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

    image_path = args.image
    prompt = args.prompt

    model_id = "vikhyatk/moondream1"
    tokenizer = Tokenizer.from_pretrained(model_id)
    moondream = Moondream.from_pretrained(model_id).to(device=device, dtype=dtype)
    moondream.eval()

    image = Image.open(image_path)
    image_embeds = moondream.encode_image(image)

    if prompt is None:
        chat_history = ""

        while True:
            question = input("> ")

            result_queue = Queue()

            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

            # Separate direct arguments from keyword arguments
            thread_args = (image_embeds, question, tokenizer, chat_history)
            thread_kwargs = {"streamer": streamer, "result_queue": result_queue}

            thread = Thread(
                target=moondream.answer_question,
                args=thread_args,
                kwargs=thread_kwargs,
            )
            thread.start()

            buffer = ""
            for new_text in streamer:
                buffer += new_text
                if not new_text.endswith("<") and not new_text.endswith("END"):
                    print(buffer, end="", flush=True)
                    buffer = ""
            print(re.sub("<$", "", re.sub("END$", "", buffer)))

            thread.join()

            answer = result_queue.get()
            chat_history += f"Question: {question}\n\nAnswer: {answer}\n\n"
    else:
        print(">", prompt)
        answer = moondream.answer_question(image_embeds, prompt, tokenizer)
        print(answer)
