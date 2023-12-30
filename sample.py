from moondream import VisionEncoder, TextModel
from PIL import Image
from huggingface_hub import snapshot_download
import argparse
from threading import Thread
from transformers import TextIteratorStreamer
import re

model_path = snapshot_download("vikhyatk/moondream0")

vision_encoder = VisionEncoder(model_path)
text_model = TextModel(model_path)

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()

image = Image.open(args.image)
image_embeds = vision_encoder(image)

if args.interactive:
    while True:
        question = input("> ")
        print(text_model.answer_question(image_embeds, question))
        print()
else:
    suggestions = text_model.suggest_questions(image_embeds)
    for question in suggestions:
        print(">", question)

        streamer = TextIteratorStreamer(text_model.tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            image_embeds=image_embeds, question=question, streamer=streamer
        )
        thread = Thread(target=text_model.answer_question, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            if not new_text.endswith("Human"):
                print(buffer, end="", flush=True)
                buffer = ""
        print(re.sub("Human$", "", buffer))
