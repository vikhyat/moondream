import argparse
from cgitb import text
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math

from PIL import Image, ImageDraw
from tqdm import tqdm
from datasets import load_dataset
from bitsandbytes.optim import AdamW
import wandb

from .weights import load_weights_into_model
from .moondream import MoondreamModel, MoondreamConfig, text_encoder
from .text import loss as t_loss

MODEL_PATH =  "/home/user/moondream/moondream/data/model.pt"
ANSWER_EOS = "<|endoftext|>"
LR = 1e-4
EPOCHS = 1
GRAD_ACCUM_STEPS = 64


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2
    

import json
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform

        self.images = data['images']
        self.annotations = data['annotations']

        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        
        self.id_to_img = {}
        for img_info in self.images:
            self.id_to_img[img_info["id"]] = img_info
        
        self.ids = list(self.id_to_img.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        img_info = self.id_to_img[image_id]

        file_name = img_info["file_name"]
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        # 3. Gather the relevant annotations for this image
        ann_list = self.img_id_to_anns.get(image_id, [])

        boxes = []
        labels = []

        for ann in ann_list:
            # COCO-style bbox is [x_min, y_min, width, height]
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        # Convert everything into torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id])

        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
    

def main():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
        
    # wandb.init(
    #     project="moondream-ft",
    #     config={
    #         "EPOCHS": EPOCHS,
    #         "BATCH_SIZE": 1,
    #         "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
    #         "LR": LR,
    #     }
    # )
    
    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model)
    
    optimizer = AdamW(
        [
            {"params": model.text.parameters()},
        ],
        lr=LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6
    )
    
    dataset =  CocoDataset(
        annotation_file="/home/user/moondream/moondream/data/roboflow/train/_annotations.coco.json",
        img_dir="/home/user/moondream/moondream/data/roboflow/train",
    )
    
    total_steps = EPOCHS * len(dataset) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(EPOCHS):
        for sample in dataset:
            i+=1
            # with torch.no_grad():
            #     img_emb = model._run_vision_encoder(sample["image"])
            #     bos_emb = text_encoder(
            #         torch.tensor([[model.config.tokenizer.bos_id]], device=model.device),
            #         model.text,
            #     )
            #     question_tokens = model.tokenizer.encode(sample["qa"]["question"]).ids
            #     question_emb = text_encoder(
            #         torch.tensor([[question_tokens]], device=model.device),
            #         model.text,
            #     ).squeeze(0)
            #     answer_tokens = model.tokenizer.encode(sample["qa"]["answer"]).ids
            #     answer_emb = text_encoder(
            #         torch.tensor([[answer_tokens]], device=model.device),
            #         model.text,
            #     ).squeeze(0)
            #     inputs_embeds = torch.cat([bos_emb, img_emb[None], question_emb, answer_emb], dim=1)
            # loss = t_loss(
            #     inputs_embeds=inputs_embeds,
            #     w = model.text,
            #     labels = torch.tensor([[answer_tokens]], device=model.device),
            #     config = config.text
            # )
            
            # loss.backward()
            
            # if i % GRAD_ACCUM_STEPS == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()

            #     lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            #     pbar.set_postfix({"step" : i//GRAD_ACCUM_STEPS, "loss" : loss.item()})
            #     pbar.update(1)
    #             wandb.log({
    #                 "loss/train": loss.item(),
    #                 "lr": optimizer.param_groups[0]['lr']
    #             })
    # wandb.finish()
    
    
   
    

if __name__ == "__main__":
    main()