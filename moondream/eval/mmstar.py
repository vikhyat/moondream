import argparse
import datasets
import torch

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

SUFFIX = " Please answer directly with only the letter of the correct option and nothing else."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)
    model.compile()

    dataset = datasets.load_dataset("Lin-Chen/MMStar", split="val")

    correct = 0
    total = 0
    category_stats = {}

    for row in tqdm(dataset, disable=args.debug):
        image = row["image"]
        question = row["question"] + SUFFIX
        answer = row["answer"]
        model_answer = model.query(image, question)["answer"]

        category = f"{row['category']} / {row['l2_category']}"
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}

        total += 1
        category_stats[category]["total"] += 1

        if model_answer.strip().lower() == answer.strip().lower():
            correct += 1
            category_stats[category]["correct"] += 1
        elif args.debug:
            print(f"Index: {row['index']}")
            print(f"Question: {row['question']}")
            print(f"Answer: {answer}")
            print(f"Model Answer: {model_answer}")
        if args.debug:
            print(f"Correct: {correct}, Total: {total}")
            print(f"Accuracy: {correct * 100 / total:.2f}")
            print("Results by category:")
            for category, stats in category_stats.items():
                acc = stats["correct"] * 100 / stats["total"]
                print(f"{category}: {stats['correct']}/{stats['total']} = {acc:.2f}%")
            print("---------")

    print(f"Correct: {correct}, Total: {total}")
    print(f"Accuracy: {correct * 100 / total:.2f}")

    print("\nResults by category:")
    for category, stats in category_stats.items():
        acc = stats["correct"] * 100 / stats["total"]
        print(f"{category}: {stats['correct']}/{stats['total']} = {acc:.2f}%")
