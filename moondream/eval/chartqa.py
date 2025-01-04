import argparse
import datasets
import torch

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

PREFIX = "Analyze the chart carefully, consider both visual features and data values, and provide a precise answer without any additional explanation or formatting. "

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

    dataset = datasets.load_dataset("moondream/chartqa", split="test").cast_column(
        "image", datasets.Image(decode=True)
    )

    correct = 0
    total = 0

    for row in tqdm(dataset, disable=args.debug):
        image = row["image"]
        encoded_image = model.encode_image(image)

        for qa in row["qa"]:
            question = PREFIX + qa["question"]
            answer = qa["answer"]
            model_answer = model.query(encoded_image, question)["answer"]

            total += 1
            if model_answer.strip().lower() == answer.strip().lower():
                correct += 1
            elif args.debug:
                print(f"Question: {qa['question']}")
                print(f"Answer: {answer}")
                print(f"Model Answer: {model_answer}")
            if args.debug:
                print(f"Correct: {correct}, Total: {total}")
                print(f"Accuracy: {correct * 100 / total:.2f}")
                print("---------")

    print(f"Correct: {correct}, Total: {total}")
    print(f"Accuracy: {correct * 100 / total:.2f}")
