import argparse
import datasets
import torch

from tqdm import tqdm
import json

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

PREFIX = "Analyze the chart carefully, consider both visual features and data values, and provide a precise answer without any additional explanation or formatting. "


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(
    target: str, prediction: str, max_relative_change: float = 0.05
) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction = str(prediction)
    target = str(target)
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction == target


def eval_chartqa(model, debug=False):
    dataset = datasets.load_dataset("vikhyatk/chartqa", split="test")

    correct = 0
    total = 0
    human_correct = 0
    human_total = 0

    for row in tqdm(dataset, disable=debug):
        image = row["image"]
        encoded_image = model.encode_image(image)

        for qa in row["qa"]:
            question = PREFIX + qa["question"]
            answer = qa["answer"]
            model_answer = model.query(encoded_image, question)["answer"]

            # Attempt to parse both answers into lists, otherwise
            try:
                answer_list = json.loads(answer)
                model_answer_list = json.loads(model_answer)
                if not (
                    isinstance(answer_list, list)
                    and isinstance(model_answer_list, list)
                    and len(answer_list) == len(model_answer_list)
                ):
                    raise ValueError
            except:
                # If parsing fails or lengths are not equal, compare the strings directly instead
                answer_list = [answer]
                model_answer_list = [model_answer]

            total += 1
            if qa["source"] == "human":
                human_total += 1

            if all(
                relaxed_correctness(
                    str(cur_answer).strip().lower(),
                    str(cur_model_answer).strip().lower(),
                )
                for cur_answer, cur_model_answer in zip(answer_list, model_answer_list)
            ):
                correct += 1
                if qa["source"] == "human":
                    human_correct += 1
            elif debug:
                print(f"Question: {qa['question']}")
                print(f"Answer: {answer}")
                print(f"Model Answer: {model_answer}")
            if debug:
                print(
                    f"Correct: {correct}, Total: {total}, Human Correct: {human_correct}, Human Total: {human_total}"
                )
                print(f"Human Accuracy: {human_correct * 100 / human_total:.2f}")
                print(f"Total Accuracy: {correct * 100 / total:.2f}")
                print("---------")

    return {
        "human_acc": human_correct * 100 / human_total,
        "total_acc": correct * 100 / total,
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

    results = eval_chartqa(model, args.debug)
    print(f"Human Accuracy: {results['human_acc']:.2f}")
    print(f"Total Accuracy: {results['total_acc']:.2f}")
