import json
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
from safetensors.torch import save_file

from PIL import Image
from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit
import wandb
import random

from ..torch.weights import load_weights_into_model, RegionModel
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder
from ..torch.text import _produce_hidden
from ..torch.region import (
    decode_coordinate,
    decode_size,
    encode_coordinate,
    encode_size,
)

MODEL_PATH = "/home/user/moondream/moondream/data/model.safetensors"
ANSWER_EOS = "<|endoftext|>"
LR = 5e-5
EPOCHS = 1
GRAD_ACCUM_STEPS = 64


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


def region_loss(
    hidden_states: torch.Tensor,
    w: RegionModel,
    labels: torch.Tensor,
    c_idx: torch.Tensor,
    s_idx: torch.Tensor,
):
    l_idx = torch.arange(len(labels))

    c_idx = c_idx - 1
    c_hidden = hidden_states[:, c_idx, :]
    c_logits = decode_coordinate(c_hidden, w)
    c_labels = labels[(l_idx % 4) < 2]

    c_loss = F.cross_entropy(
        c_logits.view(-1, c_logits.size(-1)),
        c_labels,
    )

    s_idx = s_idx - 1
    s_hidden = hidden_states[:, s_idx, :]
    s_logits = decode_size(s_hidden, w).view(-1, 1024)
    s_labels = labels[(l_idx % 4) >= 2]

    s_loss = F.cross_entropy(s_logits, s_labels)

    return c_loss + s_loss


class CocoDataset(Dataset):
    """
    Dataset class for COCO type data.
    To download the Roboflow railwayvision dataset visit: https://universe.roboflow.com/research-zwl99/railwayvision
    """

    def __init__(self, annotation_file, img_dir, transform=None):
        self.annotation_file = annotation_file
        self.img_dir = img_dir
        self.transform = transform

        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]

        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        self.ids = []
        self.id_to_img = {}
        for img_info in self.images:
            img_id = img_info["id"]
            if img_id in self.img_id_to_anns and len(self.img_id_to_anns[img_id]) > 0:
                self.ids.append(img_id)
                self.id_to_img[img_id] = img_info
        random.shuffle(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.id_to_img[image_id]

        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        ann_list = self.img_id_to_anns[image_id]

        boxes = []

        for ann in ann_list:
            bbox = ann["bbox"]
            boxes.append(
                [
                    (bbox[0] + (bbox[2] / 2)) / width,
                    (bbox[1] + (bbox[3] / 2)) / height,
                    bbox[2] / width,
                    bbox[3] / height,
                ]
            )

        boxes = torch.as_tensor(boxes, dtype=torch.float16)

        return {
            "image": image,
            "boxes": boxes,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }


def main():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project="moondream-ft",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        },
    )

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model)

    optimizer = AdamW8bit(
        [
            {"params": model.region.parameters()},
        ],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    # Add path to annotation file and img dir
    dataset = CocoDataset(
        annotation_file="/home/user/moondream/moondream/data/roboflow/train/_annotations.coco.json",
        img_dir="/home/user/moondream/moondream/data/roboflow/train",
    )

    total_steps = EPOCHS * len(dataset) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(EPOCHS):
        for sample in dataset:
            with torch.no_grad():
                i += 1
                img_emb = model._run_vision_encoder(sample["image"])
                bos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.bos_id]], device=model.device
                    ),
                    model.text,
                )

                # Basic prompt to detect a crack in the railway tracks
                instruction = "\n\nDetect: crack\n\n"
                instruction_tokens = model.tokenizer.encode(instruction).ids
                instruction_emb = text_encoder(
                    torch.tensor([[instruction_tokens]], device=model.device),
                    model.text,
                ).squeeze(0)

                eos_token = model.tokenizer.encode(ANSWER_EOS).ids
                eos_emb = text_encoder(
                    torch.tensor([eos_token], device=model.device),
                    model.text,
                )

                cs_emb = []
                cs_labels = []
                c_idx = []
                s_idx = []
                if len(sample["boxes"]) > 1:
                    pass

                for bb in sample["boxes"]:
                    l_cs = len(cs_emb)
                    cs_emb.extend(
                        [
                            encode_coordinate(bb[0].unsqueeze(0), model.region),
                            encode_coordinate(bb[1].unsqueeze(0), model.region),
                            encode_size(bb[2:4], model.region),
                        ]
                    )
                    c_idx.extend([l_cs, l_cs + 1])
                    s_idx.append(l_cs + 2)
                    cs_labels.extend(
                        [min(max(torch.round(p * 1023), 0), 1023) for p in bb]
                    )

                cs_emb = torch.stack(cs_emb)

                inputs_embeds = torch.cat(
                    [bos_emb, img_emb[None], cs_emb[None], eos_emb], dim=1
                )
                prefix = inputs_embeds.size(1) - cs_emb.size(0)
                c_idx = torch.tensor(c_idx) + prefix
                s_idx = torch.tensor(s_idx) + prefix

            hidden = _produce_hidden(
                inputs_embeds=inputs_embeds, w=model.text, config=config.text
            )

            loss = region_loss(
                hidden_states=hidden,
                w=model.region,
                labels=torch.stack(cs_labels).to(torch.int64),
                c_idx=c_idx,
                s_idx=s_idx,
            )

            loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                pbar.set_postfix({"step": i // GRAD_ACCUM_STEPS, "loss": loss.item()})
                pbar.update(1)
                wandb.log(
                    {"loss/train": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
                )
    wandb.finish()
    # Add save path: ex. home/model.safetensors
    save_file(model.state_dict(), "")


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_region
    """
    main()
