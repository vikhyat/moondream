import torch
import argparse

from .weights import load_weights_into_model
from .hf_moondream import HfConfig, HfMoondream

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="vikhyatk/moondream-next")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    config = HfConfig()
    model = HfMoondream(config)
    load_weights_into_model(args.ckpt, model.model)
    model = model.to(dtype=torch.float16)

    model.push_to_hub(args.model_name, config=config)
