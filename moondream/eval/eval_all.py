import argparse
import torch

from pprint import pprint

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

from .countbenchqa import eval_countbenchqa
from .pope import evaluate_pope
from .realworldqa import eval_realworldqa
from .chartqa import eval_chartqa
from .textvqa import eval_textvqa
from .docvqa import eval_docvqa
from .mmstar import eval_mmstar
from .coco_map import eval_coco_map
from .naturalbench import eval_naturalbench
from .tallyqa import eval_tallyqa


def create_model(ckpt_path):
    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(ckpt_path, model)
    model.compile()
    return model


def eval_all(model, skip=[]):
    evals = {
        "countbenchqa": eval_countbenchqa,
        "pope": evaluate_pope,
        "realworldqa": eval_realworldqa,
        "chartqa": eval_chartqa,
        "mmstar": eval_mmstar,
        "docvqa": eval_docvqa,
        "coco_map": eval_coco_map,
        "textvqa": eval_textvqa,
        "naturalbench": eval_naturalbench,
        "tallyqa": eval_tallyqa,
    }

    for b in skip:
        del evals[b]

    results = {}
    for name, eval_fn in evals.items():
        results[name] = eval_fn(model)
        pprint({k: v for k, v in results[name].items() if k != "results"})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    model = create_model(args.model)
    eval_all(model)
