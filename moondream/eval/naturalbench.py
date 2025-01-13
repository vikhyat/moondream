from datasets import load_dataset
from tqdm import tqdm
import torch

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model


def eval_naturalbench(model, debug=False):
    # Yes, the benchmark test set is stored in the 'train' split...
    dataset = load_dataset("BaiqiL/NaturalBench", split="train")

    acc = []
    q_acc = []
    i_acc = []
    g_acc = []

    for row in tqdm(dataset, disable=debug):
        if row["Question_Type"] == "yes_no":
            suffix = " Answer yes or no."
        else:
            suffix = ""

        images = [row["Image_0"], row["Image_1"], row["Image_0"], row["Image_1"]]
        prompts = [
            row["Question_0"] + suffix,
            row["Question_0"] + suffix,
            row["Question_1"] + suffix,
            row["Question_1"] + suffix,
        ]
        expected = [
            row["Image_0_Question_0"].strip().lower(),
            row["Image_1_Question_0"].strip().lower(),
            row["Image_0_Question_1"].strip().lower(),
            row["Image_0_Question_1"].strip().lower(),
        ]

        answers = []
        for img, prompt in zip(images, prompts):
            encoded_image = model.encode_image(img)
            answer = model.query(encoded_image, prompt)["answer"]
            answers.append(answer.strip().lower())

        if debug:
            for i, (q, a, e) in enumerate(zip(prompts, answers, expected)):
                print(f"Q{i}: {q}")
                print(f"Model: {a}")
                print(f"Expected: {e}")
                print(f"Correct: {a == e}")
                print("---")

        acc.append(answers[0] == expected[0])
        acc.append(answers[1] == expected[1])
        acc.append(answers[2] == expected[2])
        acc.append(answers[3] == expected[3])

        i_acc.append(answers[0] == expected[0] and answers[2] == expected[2])
        i_acc.append(answers[1] == expected[1] and answers[3] == expected[3])

        q_acc.append(answers[0] == expected[0] and answers[1] == expected[1])
        q_acc.append(answers[2] == expected[2] and answers[3] == expected[3])

        g_acc.append(
            answers[0] == expected[0]
            and answers[1] == expected[1]
            and answers[2] == expected[2]
            and answers[3] == expected[3]
        )

        if debug:
            print(f"Current Overall Accuracy: {sum(acc) / len(acc):.4f}")
            print(f"Current Image Accuracy: {sum(i_acc) / len(i_acc):.4f}")
            print(f"Current Question Accuracy: {sum(q_acc) / len(q_acc):.4f}")
            print(f"Current Group Accuracy: {sum(g_acc) / len(g_acc):.4f}")
            print("=========")

    return {
        "overall_acc": sum(acc) / len(acc),
        "image_acc": sum(i_acc) / len(i_acc),
        "question_acc": sum(q_acc) / len(q_acc),
        "group_acc": sum(g_acc) / len(g_acc),
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

    results = eval_naturalbench(model, debug=args.debug)

    print(f"Overall Accuracy: {results['overall_acc']:.4f}")
    print(f"Image Accuracy: {results['image_acc']:.4f}")
    print(f"Question Accuracy: {results['question_acc']:.4f}")
    print(f"Group Accuracy: {results['group_acc']:.4f}")
