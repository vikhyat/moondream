from moondream import VisionEncoder, TextModel
from PIL import Image
import argparse

vision_encoder = VisionEncoder()
text_model = TextModel()

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
