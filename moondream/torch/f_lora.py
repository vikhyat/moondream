import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil


from pathlib import Path
from urllib.request import Request, urlopen


from .utils import hf_hub_dir, rename_state_dict


LORA_ENDPOINT = "https://api-staging.moondream.ai/v1/variants/{lora_id}/download"
LORA_KEYWORDS = ["mlp", "attn"]
SCALE = 8 / 16


def replace_with_lora_linear(model: nn.Module, keywords=LORA_KEYWORDS):
    # TODO: change to just loop through named params
    named_mods = {name: mod for name, mod in model.named_modules()}
    for name, mod in named_mods.items():
        if isinstance(mod, nn.Linear) and (
            not keywords or any(k in name for k in keywords)
        ):

            new_layer = LoRALinear(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                dtype=mod.weight.dtype,
                device=mod.weight.device,
            )
            new_layer.weight = mod.weight
            new_layer.bias = mod.bias

            parent = model
            *path, leaf = name.split(".")
            for attr in path:
                parent = getattr(parent, attr)
            setattr(parent, leaf, new_layer)


class LoRALinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, **kw):
        super().__init__(in_features, out_features, bias=bias, **kw)

        self.register_buffer(
            "lora_delta",
            torch.zeros_like(self.weight, dtype=self.weight.dtype),
            persistent=False,
        )

        self.register_buffer(
            "_lora_active", torch.tensor(0, dtype=torch.bool), persistent=False
        )

    def set_lora(self, delta: torch.Tensor | None):
        if delta is None:
            self._lora_active.zero_()
        else:
            if delta.shape != self.weight.shape:
                raise ValueError("LoRA delta shape mismatch")
            self.lora_delta = delta
            self._lora_active.fill_(True)

    def forward(self, x):
        if self._lora_active:
            return F.linear(
                x,
                self.weight + self.lora_delta.to(self.weight.device),
                self.bias.to(self.weight.device),
            )
        return F.linear(x, self.weight, self.bias)


class LoRAPool:

    def __init__(self, model: nn.Module, device):
        self.model = model
        self.device = device
        self._cpu, self._gpu = {}, {}

        self._layers = {
            name: mod
            for name, mod in model.named_modules()
            if isinstance(mod, LoRALinear)
        }

    def activate(self, lora_id: str | None):
        for m in self._layers.values():
            m.set_lora(None)

        if lora_id is None:
            return

        delta_map = self._get_gpu(lora_id)
        for name, delta in delta_map.items():
            self._layers[name].set_lora(delta)

    def _get_gpu(self, lora_id):
        if lora_id in self._gpu:
            return self._gpu[lora_id]

        if lora_id not in self._cpu:
            self._cpu[lora_id] = self._load_delta_cpu(lora_id)

        self._gpu[lora_id] = {
            n: t.to(self.device, non_blocking=True)
            for n, t in self._cpu[lora_id].items()
        }
        return self._gpu[lora_id]

    def _load_delta_cpu(self, lora_id):
        sd = rename_state_dict(
            torch.load(self._retrieve_lora(lora_id), map_location="cpu")
        )
        scale = sd.get("scale", SCALE)
        out = {}
        for prefix in {k.rsplit(".", 1)[0] for k in sd.keys()}:
            A = sd[f"{prefix}.A"]
            B = sd[f"{prefix}.B"]
            out[prefix] = (B @ A) * scale
        return out

    def _retrieve_lora(self, lora_id) -> Path:
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
