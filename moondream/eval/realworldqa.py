import argparse
import datasets
import torch

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model


def eval_realworldqa(model, debug=False):
    dataset = datasets.load_dataset("lmms-lab/RealWorldQA", split="test")

    correct = 0
    total = 0

    for row in tqdm(dataset, disable=debug):
        image = row["image"]
        question = row["question"]
        answer = row["answer"]
        model_answer = model.query(image, question)["answer"]

        total += 1
        if model_answer.strip().lower() == answer.strip().lower():
            correct += 1
        elif debug:
            print(f"Image: {row['image_path']}")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Model Answer: {model_answer}")
        if debug:
            print(f"Correct: {correct}, Total: {total}")
            print(f"Accuracy: {correct * 100 / total:.2f}")
            print("---------")

    return {
        "acc": correct * 100 / total,
        "correct_count": correct,
        "total_count": total,
    }


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

    result = eval_realworldqa(model, args.debug)

    print(f"Accuracy: {result['acc']:.2f}")
    print(f"Correct: {result['correct_count']} / {result['total_count']}")
