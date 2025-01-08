import argparse
import datasets
import torch

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model
from .utils import VQAScorer

PREFIX_TEXTVQA = "Read the text in the image and provide a brief lowercase answer. Respond 'unanswerable' only if there is no plausible answer. "

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

    dataset = datasets.load_dataset("vikhyatk/textvqa_val", split="validation")
    scorer = VQAScorer()

    total_score = 0
    total_samples = 0

    for row in tqdm(dataset, disable=args.debug):
        image = row["image"]
        encoded_image = model.encode_image(image)
        question = PREFIX_TEXTVQA + row["question"]
        model_answer = model.query(encoded_image, question)["answer"]

        score = scorer.compute_score(model_answer, row["answers"])
        total_score += score
        total_samples += 1

        if args.debug:
            print(f"Question: {row['question']}")
            print(f"Ground Truth Answers: {row['answers']}")
            print(f"Model Answer: {model_answer}")
            print(f"Score: {score}")
            print(f"Running Average Score: {total_score * 100 / total_samples:.2f}")
            print("---------")

    final_score = total_score * 100 / total_samples
    print(f"TextVQA Score: {final_score:.2f}")
