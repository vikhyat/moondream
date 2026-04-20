"""Simple query finetuning example for SSV2 3x3 action recognition.

Dataset: moondream/ssv2-3x3

Requires:
    pip install datasets pillow

To run with a local moondream-python checkout:
    PYTHONPATH=/path/to/moondream-python python train_ssv2_3x3_query.py

Set MOONDREAM_API_KEY.
Optional: HF_TOKEN.
"""

import io
import os
import re
import time
from collections import Counter

from datasets import load_dataset
from PIL import Image

import moondream as md

QUESTION = "This is a 3x3 grid of frames from a video. What action is happening?"

STEPS = 20
BATCH_SIZE = 128
EVAL_EVERY = 5
EVAL_LIMIT = 100
LR = 2e-4
RANK = 32
SEED = 42
MAX_TOKENS = 20

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text):
    return " ".join(_NORMALIZE_RE.sub(" ", text.lower()).split())


def token_f1(prediction, target):
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    target_counts = Counter(target_tokens)
    overlap = sum(
        min(pred_counts[token], target_counts[token])
        for token in pred_counts.keys() & target_counts.keys()
    )
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2.0 * precision * recall / (precision + recall)


def decode_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if image.get("bytes") is not None:
        with Image.open(io.BytesIO(image["bytes"])) as decoded:
            return decoded.convert("RGB")
    with Image.open(image["path"]) as decoded:
        return decoded.convert("RGB")


def load_eval_examples():
    rows = load_dataset(
        "moondream/ssv2-3x3",
        split="test",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    ).shuffle(seed=SEED, buffer_size=1000)

    examples = []
    for row in rows:
        row["image"] = decode_image(row["image"])
        examples.append(row)
        if len(examples) == EVAL_LIMIT:
            break
    return examples


def train_groups():
    epoch = 0
    while True:
        rows = load_dataset(
            "moondream/ssv2-3x3",
            split="train",
            streaming=True,
            token=os.environ.get("HF_TOKEN"),
        ).shuffle(seed=SEED + epoch, buffer_size=1000)
        epoch += 1

        batch = []
        for row in rows:
            batch.append({
                "mode": "sft",
                "request": {
                    "skill": "query",
                    "image": decode_image(row["image"]),
                    "question": QUESTION,
                    "settings": {"temperature": 0.0, "max_tokens": MAX_TOKENS},
                },
                "target": {"answer": row["annotation_text"]},
            })
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []

        if batch:
            yield batch


def evaluate(ft, examples):
    exact_matches = 0
    token_f1_sum = 0.0
    for example in examples:
        answer = ft.rollouts(
            "query",
            image=example["image"],
            question=QUESTION,
            settings={"temperature": 0.0, "max_tokens": MAX_TOKENS},
        )["rollouts"][0]["output"]["answer"]
        exact_matches += normalize_text(answer) == normalize_text(example["annotation_text"])
        token_f1_sum += token_f1(answer, example["annotation_text"])
    total = len(examples)
    return exact_matches / total, token_f1_sum / total


def main():
    eval_examples = load_eval_examples()

    ft = md.ft(
        api_key=os.environ["MOONDREAM_API_KEY"],
        name=f"ssv2-3x3-query-{int(time.time())}",
        rank=RANK,
    )
    print(f"Created finetune: {ft.finetune_id} ({ft.name})", flush=True)

    for groups in train_groups():
        step = ft.train_step(groups, lr=LR)
        print(f"step={step['step']} sft_loss={step['sft_loss']:.4f}", flush=True)

        if step["step"] % EVAL_EVERY == 0 or step["step"] == STEPS:
            exact_match, mean_token_f1 = evaluate(ft, eval_examples)
            ft.log_metrics(
                step=step["step"],
                metrics={
                    "eval/exact_match": exact_match,
                    "eval/token_f1": mean_token_f1,
                },
            )
            print(
                f"eval step={step['step']} exact_match={exact_match:.3f} token_f1={mean_token_f1:.3f}",
                flush=True,
            )

        if step["step"] == STEPS:
            break

    checkpoint = ft.save_checkpoint()["checkpoint"]
    print(f"Saved checkpoint: {checkpoint['checkpoint_id']}", flush=True)
    print(f"Model ID: {ft.model(checkpoint['step'])}", flush=True)


if __name__ == "__main__":
    main()
