import torch
from safetensors import safe_open

torch.set_default_device("cpu")
tensors = {}
with safe_open(
    "/ephemeral/VLMEvalKit/weights/moondream-01-08-2025.safetensors",
    framework="pt",
    device=0,
) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

pass
