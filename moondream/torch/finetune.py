import argparse
from cgitb import text
import json
import os
import torch

from PIL import Image, ImageDraw
from tqdm import tqdm

from .weights import load_weights_into_model
from .moondream import MoondreamModel, MoondreamConfig, text_encoder
from .text import loss as t_loss

MODEL_PATH =  "/workspace/moondream/moondream/data/model.pt"
IMAGE_PATH = "/workspace/moondream/moondream/data/500_ftqd.jpg"

def main():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
    
    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model)
    
    img = Image.open(IMAGE_PATH)
    prompt = "Caption this image. The image contains a porsche 911 gt3rs"
    with torch.no_grad():
        img_emb = model._run_vision_encoder(img)
        bos_emb = text_encoder(
            torch.tensor([[model.config.tokenizer.bos_id]], device=model.device),
            model.text,
        )
        prompt_tokens = model.tokenizer.encode(prompt).ids
        prompt_emb = text_encoder(
            torch.tensor([[prompt_tokens]], device=model.device),
            model.text,
        ).squeeze(0)
        inputs_embeds = torch.cat([bos_emb, img_emb[None], prompt_emb], dim=1)
    loss = t_loss(
        inputs_embeds=inputs_embeds,
        w = model.text,
        labels = torch.tensor([[prompt_tokens]], device=model.device),
        config = config.text
    )
    pass
    
    
   
    

if __name__ == "__main__":
    main()