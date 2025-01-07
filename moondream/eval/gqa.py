import argparse
import datasets
import torch

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model


def evaluate_gqa_answer(prediction: str, ground_truth: str) -> bool:
    """
    Evaluates if a predicted answer matches the ground truth using GQA evaluation rules.

    Args:
        prediction: Model's predicted answer string
        ground_truth: Ground truth answer string

    Returns:
        bool: True if answers match after preprocessing, False otherwise
    """
    # Preprocess prediction
    pred = prediction.strip().lower()
    pred = pred.split(".")[0]
    pred = pred.split(",")[0]
    pred = pred.split("!")[0]

    # Remove common prefixes from prediction
    for prefix in ["is ", "are ", "a ", "an ", "the "]:
        if pred.startswith(prefix):
            pred = pred[len(prefix) :]

    # Remove " of" suffix and anything after from prediction
    if " of" in pred:
        pred = pred.split(" of")[0]
    pred = pred.strip()

    # Preprocess ground truth the same way
    truth = ground_truth.strip().lower()
    truth = truth.split(".")[0]
    truth = truth.split(",")[0]
    truth = truth.split("!")[0]

    for prefix in ["is ", "are ", "a ", "an ", "the "]:
        if truth.startswith(prefix):
            truth = truth[len(prefix) :]

    if " of" in truth:
        truth = truth.split(" of")[0]
    truth = truth.strip()

    return pred == truth


PREFIX = "Consider both visual features and relationships, and think carefully before providing the final answer. "

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

    dataset = datasets.load_dataset("vikhyatk/gqa-val", split="test")

    total = 0
    correct = 0

    for row in tqdm(dataset, disable=args.debug):
        image = row["image"]
        encoded_image = model.encode_image(image)

        for qa in row["qa"]:
            question = PREFIX + qa["question"]
            answer = qa["answer"]

            model_answer = model.query(encoded_image, question)["answer"]

            total += 1
            if evaluate_gqa_answer(model_answer, answer):
                correct += 1
            elif args.debug:
                print(f"Question: {qa['question']}")
                print(f"Answer: {answer}")
                print(f"Model Answer: {model_answer}")
                print(f"Correct: {correct}, Total: {total}")
                print(f"Accuracy: {correct * 100 / total:.2f}")
                print("---------")

    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct * 100 / total:.2f}")
