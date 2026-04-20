"""Simple query finetuning example for GeoGuessr country classification.

Dataset: moondream/geoguessr-countries-finetune

Requires:
    pip install datasets pillow

To run with a local moondream-python checkout:
    PYTHONPATH=/path/to/moondream-python python train_geoguessr_countries_query.py

Set MOONDREAM_API_KEY.
Optional: HF_TOKEN.
"""

import io
import os
import re
import time
import unicodedata

from datasets import load_dataset
from PIL import Image

import moondream as md

QUESTION = "What country is this, return only the name."

STEPS = 20
BATCH_SIZE = 16
EVAL_EVERY = 5
EVAL_LIMIT = 100
LR = 2e-5
RANK = 8
SEED = 42
MAX_TOKENS = 10

COUNTRY_ALIASES = {
    "czech republic": "czechia",
    "holland": "netherlands",
    "the netherlands": "netherlands",
    "russian federation": "russia",
    "republic of korea": "south korea",
    "turkiye": "turkey",
    "britain": "united kingdom",
    "england": "united kingdom",
    "great britain": "united kingdom",
    "u k": "united kingdom",
    "uk": "united kingdom",
    "america": "united states",
    "u s": "united states",
    "u s a": "united states",
    "united states of america": "united states",
    "us": "united states",
    "usa": "united states",
}


def normalize_country(text):
    country = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    country = " ".join(re.findall(r"[a-z0-9]+", country.lower()))
    if country.startswith("the "):
        country = country[4:]
    return COUNTRY_ALIASES.get(country, country)


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
        "moondream/geoguessr-countries-finetune",
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
            "moondream/geoguessr-countries-finetune",
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
                "target": {"answer": row["country"]},
            })
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []

        if batch:
            yield batch


def evaluate(ft, examples):
    correct = 0
    for example in examples:
        answer = ft.rollouts(
            "query",
            image=example["image"],
            question=QUESTION,
            settings={"temperature": 0.0, "max_tokens": MAX_TOKENS},
        )["rollouts"][0]["output"]["answer"]
        correct += normalize_country(answer) == normalize_country(example["country"])
    return correct / len(examples)


def main():
    eval_examples = load_eval_examples()

    ft = md.ft(
        api_key=os.environ["MOONDREAM_API_KEY"],
        name=f"geoguessr-countries-query-{int(time.time())}",
        rank=RANK,
    )
    print(f"Created finetune: {ft.finetune_id} ({ft.name})", flush=True)

    for groups in train_groups():
        step = ft.train_step(groups, lr=LR)
        print(f"step={step['step']} sft_loss={step['sft_loss']:.4f}", flush=True)

        if step["step"] % EVAL_EVERY == 0 or step["step"] == STEPS:
            country_match = evaluate(ft, eval_examples)
            ft.log_metrics(
                step=step["step"],
                metrics={"eval/country_match": country_match},
            )
            print(
                f"eval step={step['step']} country_match={country_match:.3f}",
                flush=True,
            )

        if step["step"] == STEPS:
            break

    checkpoint = ft.save_checkpoint()["checkpoint"]
    print(f"Saved checkpoint: {checkpoint['checkpoint_id']}", flush=True)
    print(f"Model ID: {ft.model(checkpoint['step'])}", flush=True)


if __name__ == "__main__":
    main()
