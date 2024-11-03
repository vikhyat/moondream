import torch

from moondream.hf import Moondream
from moondream.hf.configuration_moondream import MoondreamConfig

MoondreamConfig.register_for_auto_class()
Moondream.register_for_auto_class("AutoModelForCausalLM")

OUT_MODEL = "vikhyatk/moondream-next"
CKPT_DIRS = []


def get_ckpt(filename):
    ckpts = [torch.load(f"{dir}/{filename}", map_location="cpu") for dir in CKPT_DIRS]
    avg_ckpt = {key: sum(ckpt[key] for ckpt in ckpts) / len(ckpts) for key in ckpts[0]}
    return avg_ckpt


config = MoondreamConfig()
model = Moondream(config)
model.vision_encoder.encoder.load_state_dict(get_ckpt("vision_encoder.final.pt"))
model.vision_encoder.projection.load_state_dict(get_ckpt("vision_projection.final.pt"))
model.text_model.load_state_dict(get_ckpt("text_model.final.pt"))
model.region_model.load_state_dict(get_ckpt("region_model.final.pt"))
model = model.to(dtype=torch.float16)

model.push_to_hub(OUT_MODEL, config=config)
