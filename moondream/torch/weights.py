import safetensors
import torch
import torch.nn as nn

from contextlib import contextmanager
from typing import Callable, List


@contextmanager
def safetensors_open(safetensors_file: str):
    """
    Simplify interfacing with safetensors files. Eliminates the need to ignore
    type errors when using the `safe_open` function.
    """
    with safetensors.safe_open(
        safetensors_file, framework="pt"
    ) as st:  # pyright: ignore

        def get_tensor(name: str) -> torch.Tensor:
            return st.get_tensor(name)

        def get_keys() -> List[str]:
            return st.keys()

        get_tensor.keys = get_keys

        yield get_tensor


def _load_weights(get_tensor: Callable[[str], torch.Tensor], model: nn.Module) -> None:
    """Internal function to load weights using a tensor getter function."""
    model = model.to(dtype=torch.float16)

    vision = model.vision
    region = model.region
    weight_map = {
        "vision_encoder.encoder.model.visual.patch_embed.linear.weight": vision[
            "patch_emb"
        ].weight,
        "vision_encoder.encoder.model.visual.patch_embed.linear.bias": vision[
            "patch_emb"
        ].bias,
        "vision_encoder.encoder.model.visual.pos_embed": vision.pos_emb,
        "vision_encoder.encoder.model.visual.norm.weight": vision["post_ln"].weight,
        "vision_encoder.encoder.model.visual.norm.bias": vision["post_ln"].bias,
        "vision_encoder.projection.mlp.fc1.weight": vision["proj_mlp"]["fc1"].weight,
        "vision_encoder.projection.mlp.fc1.bias": vision["proj_mlp"]["fc1"].bias,
        "vision_encoder.projection.mlp.fc2.weight": vision["proj_mlp"]["fc2"].weight,
        "vision_encoder.projection.mlp.fc2.bias": vision["proj_mlp"]["fc2"].bias,
        "text_model.transformer.embd.wte.weight": model.text.wte,
        "text_model.lm_head.ln.weight": model.text["post_ln"].weight,
        "text_model.lm_head.ln.bias": model.text["post_ln"].bias,
        "text_model.lm_head.linear.weight": model.text["lm_head"].weight,
        "text_model.lm_head.linear.bias": model.text["lm_head"].bias,
        "region_model.coordinate_encoder.weight": region["coord_encoder"].weight,
        "region_model.coordinate_encoder.bias": region["coord_encoder"].bias,
        "region_model.coordinate_decoder.fc1.weight": region["coord_decoder"][
            "fc1"
        ].weight,
        "region_model.coordinate_decoder.fc1.bias": region["coord_decoder"]["fc1"].bias,
        "region_model.coordinate_decoder.fc2.weight": region["coord_decoder"][
            "fc2"
        ].weight,
        "region_model.coordinate_decoder.fc2.bias": region["coord_decoder"]["fc2"].bias,
        "region_model.size_encoder.weight": region["size_encoder"].weight,
        "region_model.size_encoder.bias": region["size_encoder"].bias,
        "region_model.size_decoder.fc1.weight": region["size_decoder"]["fc1"].weight,
        "region_model.size_decoder.fc1.bias": region["size_decoder"]["fc1"].bias,
        "region_model.size_decoder.fc2.weight": region["size_decoder"]["fc2"].weight,
        "region_model.size_decoder.fc2.bias": region["size_decoder"]["fc2"].bias,
    }

    for i in range(len(model.vision["blocks"])):
        prefix = f"vision_encoder.encoder.model.visual.blocks.{i}"
        blk = model.vision["blocks"][i]
        weight_map.update(
            {
                f"{prefix}.norm1.weight": blk["ln1"].weight,
                f"{prefix}.norm1.bias": blk["ln1"].bias,
                f"{prefix}.norm2.weight": blk["ln2"].weight,
                f"{prefix}.norm2.bias": blk["ln2"].bias,
                f"{prefix}.attn.qkv.weight": blk["attn"]["qkv"].weight,
                f"{prefix}.attn.qkv.bias": blk["attn"]["qkv"].bias,
                f"{prefix}.attn.proj.weight": blk["attn"]["proj"].weight,
                f"{prefix}.attn.proj.bias": blk["attn"]["proj"].bias,
                f"{prefix}.mlp.fc1.weight": blk["mlp"]["fc1"].weight,
                f"{prefix}.mlp.fc1.bias": blk["mlp"]["fc1"].bias,
                f"{prefix}.mlp.fc2.weight": blk["mlp"]["fc2"].weight,
                f"{prefix}.mlp.fc2.bias": blk["mlp"]["fc2"].bias,
            }
        )

    for i in range(len(model.text["blocks"])):
        prefix = f"text_model.transformer.h.{i}"
        blk = model.text["blocks"][i]
        weight_map.update(
            {
                f"{prefix}.ln.weight": blk["ln"].weight,
                f"{prefix}.ln.bias": blk["ln"].bias,
                f"{prefix}.mixer.Wqkv.weight": blk["attn"]["qkv"].weight,
                f"{prefix}.mixer.Wqkv.bias": blk["attn"]["qkv"].bias,
                f"{prefix}.mixer.out_proj.weight": blk["attn"]["proj"].weight,
                f"{prefix}.mixer.out_proj.bias": blk["attn"]["proj"].bias,
                f"{prefix}.mlp.fc1.weight": blk["mlp"]["fc1"].weight,
                f"{prefix}.mlp.fc1.bias": blk["mlp"]["fc1"].bias,
                f"{prefix}.mlp.fc2.weight": blk["mlp"]["fc2"].weight,
                f"{prefix}.mlp.fc2.bias": blk["mlp"]["fc2"].bias,
            }
        )

    for key, tensor in weight_map.items():
        tensor.data.copy_(get_tensor(key))

    region.coord_features.data.copy_(
        get_tensor("region_model.coordinate_features.weight").T
    )
    region.size_features.data.copy_(get_tensor("region_model.size_features.weight").T)


def load_weights_from_safetensors(weights_file: str, model: nn.Module) -> None:
    """Load weights from a safetensors file into a MoondreamModel instance."""
    with safetensors_open(weights_file) as get_tensor:
        # Wrap the get_tensor function to handle key normalization
        name_map = {k.replace("._orig_mod", ""): k for k in get_tensor.keys()}
        _load_weights(lambda x: get_tensor(name_map[x]).to(dtype=torch.float16), model)


def load_weights_from_pt(weights_file: str, model: nn.Module) -> None:
    """Load weights from a PyTorch file into a MoondreamModel instance."""
    device = str(torch.empty(0).device)
    tensors = torch.load(weights_file, map_location=device, weights_only=True)
    tensors = {
        k.replace("._orig_mod", ""): v.to(dtype=torch.float16)
        for k, v in tensors.items()
    }
    _load_weights(lambda x: tensors[x], model)


def load_weights_into_model(weights_file: str, model: nn.Module) -> None:
    """
    Load weights from either a safetensors or PyTorch file directly into a MoondreamModel instance.

    Args:
        weights_file: Path to weights file (either .safetensors or .pt)
        model: MoondreamModel instance to load weights into
    """
    if weights_file.endswith(".safetensors"):
        load_weights_from_safetensors(weights_file, model)
    else:
        load_weights_from_pt(weights_file, model)

    # Make all parameters contiguous
    for param in model.parameters():
        param.data = param.data.contiguous()
