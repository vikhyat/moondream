import torch
import argparse
from PIL import Image
from moondream import VisionEncoder, TextModel, detect_device
from huggingface_hub import snapshot_download

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

    model_path = snapshot_download("vikhyatk/moondream1")
    vision_encoder = VisionEncoder(model_path).to(device=device, dtype=dtype)
    text_model = TextModel(model_path).to(device=device, dtype=dtype)
    image = Image.open(image_path)
    image_embeds = vision_encoder(image)

    if prompt is None:
        while True:
            question = input("> ")
            print(text_model.answer_question(image_embeds, question))
    else:
        print(">", prompt)
        print(text_model.answer_question(image_embeds, prompt))
