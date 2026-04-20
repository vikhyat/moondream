"""Simple detect finetuning example for ball-holder detection.

Dataset: maxs-m87/Ball-Holder-splits-v1

Requires:
    pip install datasets pillow

To run with a local moondream-python checkout:
    PYTHONPATH=/path/to/moondream-python python train_ballholder_detect.py

Set MOONDREAM_API_KEY.
Optional: HF_TOKEN.
"""

import io
import json
import os
import time

from datasets import load_dataset
from PIL import Image

import moondream as md

DATASET_NAME = "maxs-m87/Ball-Holder-splits-v1"
OBJECT_NAME = "player with the ball"

STEPS = 100
BATCH_SIZE = 8
NUM_ROLLOUTS = 4
EVAL_EVERY = 10
LR = 2e-3
RANK = 8
SEED = 42
MAX_TOKENS = 32
MAX_OBJECTS = 1

def decode_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if image.get("bytes") is not None:
        with Image.open(io.BytesIO(image["bytes"])) as decoded:
            return decoded.convert("RGB")
    with Image.open(image["path"]) as decoded:
        return decoded.convert("RGB")


def parse_boxes(value):
    return [
        {
            "x_min": box["x_min"],
            "y_min": box["y_min"],
            "x_max": box["x_max"],
            "y_max": box["y_max"],
        }
        for box in json.loads(value or "[]")
    ]


def load_examples(split):
    rows = load_dataset(
        DATASET_NAME,
        split=split,
        token=os.environ.get("HF_TOKEN"),
    ).shuffle(seed=SEED)

    return [
        {
            "image": decode_image(row["image"]),
            "boxes": parse_boxes(row["answer_boxes"]),
        }
        for row in rows
    ]


def detect_settings(temperature):
    return {
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "max_objects": MAX_OBJECTS,
    }


def request_stream(examples):
    while True:
        for example in examples:
            yield example, {
                "skill": "detect",
                "image": example["image"],
                "object": OBJECT_NAME,
                "num_rollouts": NUM_ROLLOUTS,
                "settings": detect_settings(1.0),
                "ground_truth": {"boxes": example["boxes"]},
            }


def evaluate(ft, examples):
    miou = 0.0
    for example in examples:
        response = ft.rollouts(
            "detect",
            image=example["image"],
            object=OBJECT_NAME,
            settings=detect_settings(0.0),
            ground_truth={"boxes": example["boxes"]},
        )
        miou += response["rewards"][0]
    return miou / len(examples)


def main():
    train_examples = load_examples("train")
    eval_examples = load_examples("validation")

    ft = md.ft(
        api_key=os.environ["MOONDREAM_API_KEY"],
        name=f"ballholder-detect-{int(time.time())}",
        rank=RANK,
    )
    print(f"Created finetune: {ft.finetune_id} ({ft.name})", flush=True)

    batch = []
    for _, response in ft.rollout_stream(request_stream(train_examples)):
        batch.append({
            "mode": "rl",
            "request": response["request"],
            "rollouts": response["rollouts"],
            "rewards": response["rewards"],
        })
        if len(batch) < BATCH_SIZE:
            continue

        step = ft.train_step(batch, lr=LR)
        batch = []
        print(f"step={step['step']} reward_mean={step['reward_mean']:.4f}", flush=True)

        if step["step"] % EVAL_EVERY == 0 or step["step"] == STEPS:
            eval_miou = evaluate(ft, eval_examples)
            ft.log_metrics(step=step["step"], metrics={"eval/miou": eval_miou})
            print(f"eval step={step['step']} miou={eval_miou:.4f}", flush=True)

        if step["step"] == STEPS:
            break

    checkpoint = ft.save_checkpoint()["checkpoint"]
    print(f"Saved checkpoint: {checkpoint['checkpoint_id']}", flush=True)
    print(f"Model ID: {ft.model(checkpoint['step'])}", flush=True)


if __name__ == "__main__":
    main()
