import math
import torch
import torch.nn as nn
import shutil

from torch.nn.utils import parametrize
from urllib.request import Request, urlopen

from .utils import hf_hub_dir, rename_state_dict

LORA_ENDPOINT = "https://api-staging.moondream.ai/v1/variants/{lora_id}/download"
R = 16
ALPHA = 8
LORA_KEYWORDS = ["mlp", "attn"]


class LoRA(nn.Module):
    """
    Calculates and returns the adapted weight; intended to be used as a parametrization.
    Let: W = W + (α / r) * (B @ A)
    """

    def __init__(self, weight: torch.Tensor, r: int, alpha: int, p: float):
        super().__init__()
        out_dim, in_dim = weight.shape
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, in_dim, dtype=weight.dtype))
        self.B = nn.Parameter(torch.empty(out_dim, r, dtype=weight.dtype))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W + self.scale * (self.B @ self.A)


def add_lora(
    model: nn.Module,
    keywords: list[str] = None,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
):
    """Attach LoRA to every nn.Linear.weight whose name contains a keyword"""

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and (
            not keywords or any(k in name for k in keywords)
        ):
            if not parametrize.is_parametrized(mod, "weight"):
                parametrize.register_parametrization(
                    mod, "weight", LoRA(mod.weight, r, alpha, dropout)
                )


def setup_lora(model, lora_id: str):
    """Downloads lora, adds parametrization, and then loads the lora weights."""

    lora_dest = download_lora(lora_id)
    add_lora(model.text, keywords=LORA_KEYWORDS, r=R, alpha = ALPHA)

    sd = torch.load(lora_dest, map_location="cpu")
    model.load_state_dict(rename_state_dict(sd), strict=False, assign=True)


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