import argparse
from PIL import Image
from moondream import VisionEncoder, TextModel
from huggingface_hub import snapshot_download

def main(image_path, prompt):
    model_path = snapshot_download("vikhyatk/moondream1")
    vision_encoder = VisionEncoder(model_path)
    text_model = TextModel(model_path)
    image = Image.open(image_path)
    image_embeds = vision_encoder(image)

    if prompt is None:
        while True:
            question = input("> ")
            print(text_model.answer_question(image_embeds, question))
    else:
        print(">", prompt)
        print(text_model.answer_question(image_embeds, prompt))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=False)
    args = parser.parse_args()
    main(args.image, args.prompt)