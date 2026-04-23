import argparse
import os

import datasets
import torch
import torch.distributed as dist

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

PREFIX = "Look at the image carefully and count the objects. Answer with just a number, without any additional text. "


def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def eval_tallyqa(model, debug=False):
    world_size, rank = _dist_info()

    dataset = datasets.load_dataset(
        "vikhyatk/tallyqa-test",
        split="test",
        download_config=datasets.DownloadConfig(num_proc=16),
    )
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank, contiguous=True)

    total = 0
    total_simple = 0
    correct = 0
    correct_simple = 0

    for row in tqdm(dataset, disable=debug or rank != 0):
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
            elif debug:
                print(f"Question: {qa['question']}")
                print(f"Answer: {answer}")
                print(f"Model Answer: {model_answer}")

            if is_simple:
                total_simple += 1
                if model_answer.strip().lower() == answer.strip().lower():
                    correct_simple += 1

            if debug:
                print(f"Simple - Correct: {correct_simple}, Total: {total_simple}")
                print(f"Simple Accuracy: {correct_simple * 100 / total_simple:.2f}")
                print(f"All - Correct: {correct}, Total: {total}")
                print(f"All Accuracy: {correct * 100 / total:.2f}")
                print("---------")

    if world_size > 1:
        counts = torch.tensor(
            [total, correct, total_simple, correct_simple],
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        total, correct, total_simple, correct_simple = counts.tolist()

    return {
        "simple_acc": correct_simple * 100 / total_simple,
        "full_acc": correct * 100 / total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Multi-GPU: launch via
    #   torchrun --nproc_per_node=<N> -m moondream.eval.tallyqa --model <path>
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.set_default_device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)
    model.compile()

    result = eval_tallyqa(model, args.debug)

    _, rank = _dist_info()
    if rank == 0:
        print(f"Simple acc: {result['simple_acc']:.2f}")
        print(f"Full acc: {result['full_acc']:.2f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
