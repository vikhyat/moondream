import argparse
import json
import os
import torch

from PIL import Image, ImageDraw
from tqdm import tqdm

from torch.utils.data import Dataset
from datasets import load_dataset

from .weights import load_weights_into_model
from .moondream import MoondreamModel, MoondreamConfig


class CaptchaDataset(Dataset):
    def __init__(self, split="train"):
        self.data = load_dataset("google/docci", trust_remote_code=True)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"],  # Should be a PIL image
            "qa": [
                {
                    "question": "Describe this image.",
                    "answer": sample["description"],
                }
            ],
        }


def main():
    model_path = ""

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)

    datasets = {
        "train": CaptchaDataset("train"),
        "test": CaptchaDataset("test"),
    }

    def collate_fn(batch):
        images = [sample["image"] for sample in batch]
        images = [moondream.vision_encoder.preprocess(image) for image in images]

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)

            for qa in sample["qa"]:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False,
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}", add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)

            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images,
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )

    # Encode image.
    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = Image.open(image_path)

    encoded_image = model.encode_image(image)


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    # Load config.

    # Load model.
