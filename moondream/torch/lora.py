import functools
import os
import shutil
import torch

from pathlib import Path
from urllib.request import Request, urlopen
from typing import Optional


def variant_cache_dir():
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache is not None:
        return Path(hf_hub_cache) / "md_variants"

    hf_home = os.environ.get("HF_HOME")
    if hf_home is not None:
        return Path(hf_home) / "hub" / "md_variants"

    return Path("~/.cache/huggingface/hub").expanduser() / "md_variants"


def cached_variant_path(variant_id: str):
    variant, *rest = variant_id.split("/", 1)
    step = rest[0] if rest else "final"

    cache_dir = variant_cache_dir() / variant
    os.makedirs(cache_dir, exist_ok=True)
    dest = cache_dir / f"{step}.pt"
    if dest.exists():
        return dest

    md_endpoint = os.getenv("MOONDREAM_ENDPOINT", "https://api.moondream.ai")

    headers = {"User-Agent": "moondream-torch"}
    api_key = os.getenv("MOONDREAM_API_KEY")
    if api_key is not None:
        headers["X-Moondream-Auth"] = api_key

    req = Request(f"{md_endpoint}/v1/variants/{variant_id}/download", headers=headers)
    with urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def nest(flat):
    tree = {}
    for k, v in flat.items():
        parts = k.split(".")
        d = tree
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return tree


@functools.lru_cache(maxsize=5)
def variant_state_dict(variant_id: Optional[str] = None, device: str = "cpu"):
    if variant_id is None:
        return None

    state_dict = torch.load(
        cached_variant_path(variant_id), map_location=device, weights_only=True
    )

    # TODO: Move these into the training code that saves checkpoints...
    rename_rules = [
        ("text_model.transformer.h", "text.blocks"),
        (".mixer", ".attn"),
        (".out_proj", ".proj"),
        (".Wqkv", ".qkv"),
        (".parametrizations.weight.0", ""),
    ]
    new_state_dict = {}
    for key, tensor in state_dict.items():
        new_key = key
        for old, new in rename_rules:
            if old in new_key:
                new_key = new_key.replace(old, new)
        new_state_dict[new_key] = tensor

    return nest(new_state_dict)
