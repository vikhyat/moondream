import contextlib
import torch
import torch.nn as nn

from .utils import hf_hub_dir, rename_state_dict

R = 16
ALPHA = 8
SCALE = ALPHA / R
LORA_KEYWORDS = ["mlp", "attn"]
LORA_ENDPOINT = "https://api-staging.moondream.ai/v1/variants/{lora_id}/download"


def load_lora_state_dict(lora_id: str) -> dict[str, torch.Tensor]:
    dest = download_lora()


def download_lora(lora_id: str):
    """Download LoRA file to HF_HUB_CACHE/moondream_lora"""

    cache_dir = hf_hub_dir() / "moondream_lora" / lora_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "lora.pt"
    if dest.exists():
        return dest

    url = LORA_ENDPOINT.format(lora_id=lora_id)
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LoraDownloader/1.0)",
        },
    )
    with urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

    return dest
