import argparse
import datasets
import torch

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

PREFIX = "Look at the image carefully and count the objects. Answer with just a number, without any additional text. "

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

    dataset = datasets.load_dataset(
        "vikhyatk/tallyqa-test",
        split="test",
        download_config=datasets.DownloadConfig(num_proc=16),
    )

    total = 0
    total_simple = 0
    correct = 0
    correct_simple = 0

    for row in tqdm(dataset, disable=args.debug):
        image = row["image"]
        encoded_image = model.encode_image(image)

        for qa in row["qa"]:
            question = PREFIX + qa["question"]
            answer = str(qa["answer"])
            is_simple = qa["is_simple"]

            model_answer = model.query(encoded_image, question)["answer"]

            total += 1
            if model_answer.strip().lower() == answer.strip().lower():
                correct += 1
            elif args.debug:
                print(f"Question: {qa['question']}")
                print(f"Answer: {answer}")
                print(f"Model Answer: {model_answer}")

            if is_simple:
                total_simple += 1
                if model_answer.strip().lower() == answer.strip().lower():
                    correct_simple += 1

            if args.debug:
                print(f"Simple - Correct: {correct_simple}, Total: {total_simple}")
                print(f"Simple Accuracy: {correct_simple * 100 / total_simple:.2f}")
                print(f"All - Correct: {correct}, Total: {total}")
                print(f"All Accuracy: {correct * 100 / total:.2f}")
                print("---------")

print(
    f"Simple: {total_simple}, Correct: {correct_simple}, Accuracy: {correct_simple*100.0/total_simple:.2f}"
)
print(f"Total: {total}, Correct: {correct}, Accuracy: {correct*100.0/total:.2f}")
