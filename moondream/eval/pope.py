import argparse
from datasets import load_dataset
from tqdm import tqdm
import torch

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

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

    pope_dataset = load_dataset("vikhyatk/POPE", split="test")

    stats = {
        "random": (0, 0),
        "popular": (0, 0),
        "adversarial": (0, 0),
    }

    for row in tqdm(pope_dataset, disable=args.debug):
        image = row["image"]
        encoded_image = model.encode_image(image)
        for split in ["adversarial", "popular", "random"]:
            for qa in row[split]:
                question = qa["question"]
                answer = qa["answer"]
                prompt = f"{question}\nAnswer yes or no."
                model_answer = model.query(encoded_image, prompt)["answer"].strip()

                if args.debug:
                    print(f"Split: {split}")
                    print(f"Question: {question}")
                    print(f"Model: {model_answer}")
                    print(f"Expected: {answer}")
                    print(f"Correct: {model_answer.lower() == answer.lower()}")
                    print("---")

                if model_answer.lower() == answer.lower():
                    stats[split] = (stats[split][0] + 1, stats[split][1] + 1)
                else:
                    stats[split] = (stats[split][0], stats[split][1] + 1)

                if args.debug:
                    for s in stats:
                        if stats[s][1] > 0:
                            print(
                                f"{s.capitalize()}: {stats[s][0]}/{stats[s][1]} = {stats[s][0] * 100.0 / stats[s][1]:.2f}%"
                            )
                    print("=========")

    print(
        "Random:",
        stats["random"][0],
        "/",
        stats["random"][1],
        ":",
        stats["random"][0] * 100.0 / stats["random"][1],
    )
    print(
        "Popular:",
        stats["popular"][0],
        "/",
        stats["popular"][1],
        ":",
        stats["popular"][0] * 100.0 / stats["popular"][1],
    )
    print(
        "Adversarial:",
        stats["adversarial"][0],
        "/",
        stats["adversarial"][1],
        ":",
        stats["adversarial"][0] * 100.0 / stats["adversarial"][1],
    )
