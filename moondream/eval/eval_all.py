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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)
    model.compile()

    evals = {
        "countbenchqa": eval_countbenchqa,
        "pope": evaluate_pope,
        "realworldqa": eval_realworldqa,
        "chartqa": eval_chartqa,
        "textvqa": eval_textvqa,
        "docvqa": eval_docvqa,
        "mmstar": eval_mmstar,
    }

    results = {}
    for name, eval_fn in evals.items():
        results[name] = eval_fn(model)
        pprint(results[name])

    print("Final Results:")
    pprint(results)
