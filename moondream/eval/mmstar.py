import datasets
import torch

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

SUFFIX = " Please answer directly with only the letter of the correct option and nothing else."


def eval_mmstar(model, debug=False):
    dataset = datasets.load_dataset("Lin-Chen/MMStar", split="val")

    correct = 0
    total = 0
    category_stats = {}
    results = []

    for row in tqdm(dataset, disable=debug, desc="MMStar"):
        image = row["image"]
        question = row["question"] + SUFFIX
        answer = row["answer"]
        model_answer = model.query(image, question)["answer"]
        is_correct = model_answer.strip().lower() == answer.strip().lower()

        category = f"{row['category']} / {row['l2_category']}"
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}

        total += 1
        category_stats[category]["total"] += 1

        results.append(
            {
                "question": question,
                "ground_truth": answer,
                "model_answer": model_answer,
                "is_correct": is_correct,
                "category": category,
            }
        )

        if is_correct:
            correct += 1
            category_stats[category]["correct"] += 1
        elif debug:
            print(f"Index: {row['index']}")
            print(f"Question: {row['question']}")
            print(f"Answer: {answer}")
            print(f"Model Answer: {model_answer}")
        if debug:
            print(f"Correct: {correct}, Total: {total}")
            print(f"Accuracy: {correct * 100 / total:.2f}")
            print("Results by category:")
            for category, stats in category_stats.items():
                acc = stats["correct"] * 100 / stats["total"]
                print(f"{category}: {stats['correct']}/{stats['total']} = {acc:.2f}%")
            print("---------")

    return {
        "acc": correct * 100 / total,
        "correct_count": correct,
        "total_count": total,
        "category_stats": category_stats,
        "results": results,
    }


if __name__ == "__main__":
    import argparse

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

    result = eval_mmstar(model, args.debug)

    print(f"Correct: {result['correct_count']}, Total: {result['total_count']}")
    print(f"Accuracy: {result['acc']:.2f}")

    print("\nResults by category:")
    for category, stats in result["category_stats"].items():
        acc = stats["correct"] * 100 / stats["total"]
        print(f"{category}: {stats['correct']}/{stats['total']} = {acc:.2f}%")
