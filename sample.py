from moondream import VisionEncoder, TextModel
from PIL import Image
from huggingface_hub import snapshot_download
import argparse

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
    for suggestion in suggestions:
        print("> ", suggestion)
        print(text_model.answer_question(image_embeds, suggestion))
        print()
