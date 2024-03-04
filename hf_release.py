import torch
from moondream import Moondream
from moondream.configuration_moondream import MoondreamConfig

MoondreamConfig.register_for_auto_class()
Moondream.register_for_auto_class("AutoModelForCausalLM")

config = MoondreamConfig()
model = Moondream(config)
model.vision_encoder.encoder.load_state_dict(
    torch.load("checkpoints/vision_encoder.s9001.pt", map_location="cpu")
)
model.vision_encoder.projection.load_state_dict(
    torch.load("checkpoints/vision_projection.s9001.pt", map_location="cpu")
)
model.text_model.load_state_dict(
    torch.load("checkpoints/text_model.s9001.pt", map_location="cpu")
)
model = model.to(dtype=torch.float16)

model.push_to_hub("vikhyatk/moondream2", config=config)
