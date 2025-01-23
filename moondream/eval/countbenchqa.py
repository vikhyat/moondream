import argparse
import datasets
import torch

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

PREFIX = "Look at the image carefully and count the objects. Answer with just a number, without any additional text. "


def eval_countbenchqa(model, debug=False):
    dataset = datasets.load_dataset("vikhyatk/CountBenchQA", split="test")

    correct = 0
    total = 0

    for row in tqdm(dataset, disable=debug, desc="CountBenchQA"):
        image = row["image"]
        encoded_image = model.encode_image(image)

        question = PREFIX + row["question"]
        answer = str(row["number"])
        model_answer = model.query(encoded_image, question)["answer"]

        total += 1
        if model_answer.strip().lower() == answer.strip().lower():
            correct += 1
        elif debug:
            print(f"Question: {row['question']}")
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

    result = eval_countbenchqa(model, args.debug)

    print(f"Accuracy: {result['acc']:.2f}")
    print(f"Correct: {result['correct_count']}, Total: {result['total_count']}")
