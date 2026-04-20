"""Simple query finetuning example for glaucoma stage classification.

Dataset: moondream/glaucoma-detection

Requires:
    pip install datasets pillow

To run with a local moondream-python checkout:
    PYTHONPATH=/path/to/moondream-python python train_glaucoma_query.py

Set MOONDREAM_API_KEY.
Optional: HF_TOKEN.
"""

import os
import re
import time

from datasets import load_dataset

import moondream as md

QUESTION = (
    "What stage of glaucoma is shown in this retinal image? "
    "Respond with one of: normal, early, advanced only."
)

STEPS = 10
NUM_ROLLOUTS = 4
EVAL_EVERY = 5
EVAL_LIMIT = 100
LR = 0.001
RANK = 8
SEED = 42
MAX_TOKENS = 10

VALID_STAGES = ("normal", "early", "advanced")


def load_examples(target_split, limit=None, shuffle=False):
    rows = load_dataset(
        "moondream/glaucoma-detection",
        split=target_split,
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if shuffle:
        rows = rows.shuffle(seed=SEED, buffer_size=1000)

    examples = []
    for row in rows:
        examples.append(row)
        if limit is not None and len(examples) == limit:
            break
    return examples


def normalize_stage(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    for stage in VALID_STAGES:
        if stage in words:
            return stage
    return ""


def request_stream(examples):
    while True:
        for example in examples:
            yield example, {
                "skill": "query",
                "image": example["image"],
                "question": QUESTION,
                "num_rollouts": NUM_ROLLOUTS,
                "settings": {"temperature": 1.0, "max_tokens": MAX_TOKENS},
            }


def evaluate(ft, examples):
    correct = 0
    for example in examples:
        answer = ft.rollouts(
            "query",
            image=example["image"],
            question=QUESTION,
            settings={"temperature": 0.0, "max_tokens": MAX_TOKENS},
        )["rollouts"][0]["output"]["answer"]
        correct += normalize_stage(answer) == example["class"]
    return correct / len(examples)


def main():
    train_examples = load_examples("train", shuffle=True)
    eval_examples = load_examples("validation", limit=EVAL_LIMIT, shuffle=True)

    ft = md.ft(
        api_key=os.environ["MOONDREAM_API_KEY"],
        name=f"glaucoma-query-{int(time.time())}",
        rank=RANK,
    )
    print(f"Created finetune: {ft.finetune_id} ({ft.name})", flush=True)

    for _, (example, response) in zip(
        range(STEPS),
        ft.rollout_stream(request_stream(train_examples)),
    ):
        rewards = [
            float(normalize_stage(r["output"]["answer"]) == example["class"])
            for r in response["rollouts"]
        ]
        step = ft.train_step([{
            "mode": "rl",
            "request": response["request"],
            "rollouts": response["rollouts"],
            "rewards": rewards,
        }], lr=LR)
        reward_mean = sum(rewards) / len(rewards)
        print(
            f"step={step['step']} label={example['class']} reward_mean={reward_mean:.3f}",
            flush=True,
        )

        if step["step"] % EVAL_EVERY == 0 or step["step"] == STEPS:
            eval_accuracy = evaluate(ft, eval_examples)
            ft.log_metrics(step=step["step"], metrics={"eval/accuracy": eval_accuracy})
            print(
                f"eval step={step['step']} accuracy={eval_accuracy:.3f}",
                flush=True,
            )

    checkpoint = ft.save_checkpoint()["checkpoint"]
    print(f"Saved checkpoint: {checkpoint['checkpoint_id']}", flush=True)
    print(f"Model ID: {ft.model(checkpoint['step'])}", flush=True)


if __name__ == "__main__":
    main()
