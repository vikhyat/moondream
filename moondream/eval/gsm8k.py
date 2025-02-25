import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
import torch

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_answer(text):
    """
    Extracts the last number (integer or float) from a string.

    Args:
        text (str): The input string to parse

    Returns:
        float or int: The last number found in the string
        None: If no number is found
    """
    # Find all numbers (integers or floats) in the string
    # This regex matches integers and decimal numbers (with or without leading zeros, ignores commas )
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text.replace(",", ""))

    if not numbers:
        return None

    # Get the last number found
    last_number = numbers[-1]

    # Convert to the appropriate type (int or float)
    if "." in last_number:
        return float(last_number)
    else:
        return int(last_number)


def eval_gsm8k(model, debug=False):
    """Evaluate the model on the GSM8K dataset."""
    # Load the GSM8K test dataset
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")

    correct = 0
    total = 0
    results = []

    for row in tqdm(gsm8k_test, disable=debug, desc="GSM8K"):

        question = row["question"]

        # Extract the ground truth answer from the answer field, just the number
        gt_answer = row["answer"].split("####")[-1].strip()

        if gt_answer is None or not gt_answer:
            if debug:
                print(
                    f"Warning: Could not parse ground truth answer from: {row['answer']}"
                )
            continue

        # Encode the question for the model
        model_response = model._text_query(question)["answer"]

        model_answer = parse_answer(model_response)

        # Convert to float for comparison (handling both integers and decimals)
        try:
            gt_answer_float = float(gt_answer)
            if model_answer is not None:
                try:
                    model_answer_float = float(model_answer)
                    is_correct = abs(model_answer_float - gt_answer_float) < 1e-6
                except:
                    is_correct = False
                    print(
                        f"failed to compute model answer float: {model_answer}, slotting in large negative."
                    )
            else:
                is_correct = False
        except ValueError:
            is_correct = False

        if is_correct:
            correct += 1
        total += 1

        result = {
            "question": question,
            "ground_truth": gt_answer,
            "model_response": model_response,
            "model_answer": model_answer,
            "correct": is_correct,
        }
        results.append(result)

        if debug:
            print(f"Question: {question}")
            print(f"Ground Truth Answer: {gt_answer}")
            print(f"Model Response: {model_response}")
            print(f"Model Answer: {model_answer}")
            print(f"Correct: {is_correct}")
            print(f"Current Accuracy: {correct/total:.4f}")
            print("---------")

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
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

    # Compile omitted to make text only work
    # model.compile()

    result = eval_gsm8k(model, args.debug)

    print(f"Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
