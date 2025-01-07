import safetensors
import torch
import torch.nn as nn

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List

from .layers import AttentionWeights, LayerNormWeights, LinearWeights, MLPWeights


@dataclass
class VisionBlock:
    ln1: LayerNormWeights
    attn: AttentionWeights
    ln2: LayerNormWeights
    mlp: MLPWeights


@dataclass
class VisionModel:
    patch_emb: LinearWeights
    pos_emb: torch.Tensor
    blocks: List[VisionBlock]
    post_ln: LayerNormWeights
    proj_mlp: MLPWeights


@dataclass
class TextBlock:
    ln: LayerNormWeights
    attn: AttentionWeights
    mlp: MLPWeights


@dataclass
class TextModel:
    wte: torch.Tensor
    blocks: List[TextBlock]
    post_ln: LayerNormWeights
    lm_head: LinearWeights


@dataclass
class RegionModel:
    coord_features: torch.Tensor
    coord_encoder: LinearWeights
    coord_decoder: MLPWeights
    size_features: torch.Tensor
    size_encoder: LinearWeights
    size_decoder: MLPWeights


@dataclass
class MoondreamModel:
    vision: VisionModel
    text: TextModel
    region: RegionModel


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

    # Vision Model
    model.vision["patch_emb"].weight.data.copy_(
        get_tensor("vision_encoder.encoder.model.visual.patch_embed.linear.weight")
    )
    model.vision["patch_emb"].bias.data.copy_(
        get_tensor("vision_encoder.encoder.model.visual.patch_embed.linear.bias")
    )
    model.vision.pos_emb.data.copy_(
        get_tensor("vision_encoder.encoder.model.visual.pos_embed")
    )

    for i in range(len(model.vision["blocks"])):
        prefix = f"vision_encoder.encoder.model.visual.blocks.{i}"

        # Layer norms
        model.vision["blocks"][i]["ln1"].weight.data.copy_(
            get_tensor(f"{prefix}.norm1.weight")
        )
        model.vision["blocks"][i]["ln1"].bias.data.copy_(
            get_tensor(f"{prefix}.norm1.bias")
        )
        model.vision["blocks"][i]["ln2"].weight.data.copy_(
            get_tensor(f"{prefix}.norm2.weight")
        )
        model.vision["blocks"][i]["ln2"].bias.data.copy_(
            get_tensor(f"{prefix}.norm2.bias")
        )

        # Attention
        model.vision["blocks"][i]["attn"]["qkv"].weight.data.copy_(
            get_tensor(f"{prefix}.attn.qkv.weight")
        )
        model.vision["blocks"][i]["attn"]["qkv"].bias.data.copy_(
            get_tensor(f"{prefix}.attn.qkv.bias")
        )
        model.vision["blocks"][i]["attn"]["proj"].weight.data.copy_(
            get_tensor(f"{prefix}.attn.proj.weight")
        )
        model.vision["blocks"][i]["attn"]["proj"].bias.data.copy_(
            get_tensor(f"{prefix}.attn.proj.bias")
        )

        # MLP
        model.vision["blocks"][i]["mlp"]["fc1"].weight.data.copy_(
            get_tensor(f"{prefix}.mlp.fc1.weight")
        )
        model.vision["blocks"][i]["mlp"]["fc1"].bias.data.copy_(
            get_tensor(f"{prefix}.mlp.fc1.bias")
        )
        model.vision["blocks"][i]["mlp"]["fc2"].weight.data.copy_(
            get_tensor(f"{prefix}.mlp.fc2.weight")
        )
        model.vision["blocks"][i]["mlp"]["fc2"].bias.data.copy_(
            get_tensor(f"{prefix}.mlp.fc2.bias")
        )

    model.vision["post_ln"].weight.data.copy_(
        get_tensor("vision_encoder.encoder.model.visual.norm.weight")
    )
    model.vision["post_ln"].bias.data.copy_(
        get_tensor("vision_encoder.encoder.model.visual.norm.bias")
    )

    model.vision["proj_mlp"]["fc1"].weight.data.copy_(
        get_tensor("vision_encoder.projection.mlp.fc1.weight")
    )
    model.vision["proj_mlp"]["fc1"].bias.data.copy_(
        get_tensor("vision_encoder.projection.mlp.fc1.bias")
    )
    model.vision["proj_mlp"]["fc2"].weight.data.copy_(
        get_tensor("vision_encoder.projection.mlp.fc2.weight")
    )
    model.vision["proj_mlp"]["fc2"].bias.data.copy_(
        get_tensor("vision_encoder.projection.mlp.fc2.bias")
    )

    # Text Model
    model.text.wte.data.copy_(get_tensor("text_model.transformer.embd.wte.weight"))

    for i in range(len(model.text["blocks"])):
        prefix = f"text_model.transformer.h.{i}"

        # Layer norm
        model.text["blocks"][i]["ln"].weight.data.copy_(
            get_tensor(f"{prefix}.ln.weight")
        )
        model.text["blocks"][i]["ln"].bias.data.copy_(get_tensor(f"{prefix}.ln.bias"))

        # Attention
        model.text["blocks"][i]["attn"]["qkv"].weight.data.copy_(
            get_tensor(f"{prefix}.mixer.Wqkv.weight")
        )
        model.text["blocks"][i]["attn"]["qkv"].bias.data.copy_(
            get_tensor(f"{prefix}.mixer.Wqkv.bias")
        )
        model.text["blocks"][i]["attn"]["proj"].weight.data.copy_(
            get_tensor(f"{prefix}.mixer.out_proj.weight")
        )
        model.text["blocks"][i]["attn"]["proj"].bias.data.copy_(
            get_tensor(f"{prefix}.mixer.out_proj.bias")
        )

        # MLP
        model.text["blocks"][i]["mlp"]["fc1"].weight.data.copy_(
            get_tensor(f"{prefix}.mlp.fc1.weight")
        )
        model.text["blocks"][i]["mlp"]["fc1"].bias.data.copy_(
            get_tensor(f"{prefix}.mlp.fc1.bias")
        )
        model.text["blocks"][i]["mlp"]["fc2"].weight.data.copy_(
            get_tensor(f"{prefix}.mlp.fc2.weight")
        )
        model.text["blocks"][i]["mlp"]["fc2"].bias.data.copy_(
            get_tensor(f"{prefix}.mlp.fc2.bias")
        )

    model.text["post_ln"].weight.data.copy_(get_tensor("text_model.lm_head.ln.weight"))
    model.text["post_ln"].bias.data.copy_(get_tensor("text_model.lm_head.ln.bias"))

    model.text["lm_head"].weight.data.copy_(
        get_tensor("text_model.lm_head.linear.weight")
    )
    model.text["lm_head"].bias.data.copy_(get_tensor("text_model.lm_head.linear.bias"))

    # Region Model
    model.region.coord_features.data.copy_(
        get_tensor("region_model.coordinate_features.weight").T
    )
    model.region["coord_encoder"].weight.data.copy_(
        get_tensor("region_model.coordinate_encoder.weight")
    )
    model.region["coord_encoder"].bias.data.copy_(
        get_tensor("region_model.coordinate_encoder.bias")
    )

    model.region["coord_decoder"]["fc1"].weight.data.copy_(
        get_tensor("region_model.coordinate_decoder.fc1.weight")
    )
    model.region["coord_decoder"]["fc1"].bias.data.copy_(
        get_tensor("region_model.coordinate_decoder.fc1.bias")
    )
    model.region["coord_decoder"]["fc2"].weight.data.copy_(
        get_tensor("region_model.coordinate_decoder.fc2.weight")
    )
    model.region["coord_decoder"]["fc2"].bias.data.copy_(
        get_tensor("region_model.coordinate_decoder.fc2.bias")
    )

    model.region.size_features.data.copy_(
        get_tensor("region_model.size_features.weight").T
    )
    model.region["size_encoder"].weight.data.copy_(
        get_tensor("region_model.size_encoder.weight")
    )
    model.region["size_encoder"].bias.data.copy_(
        get_tensor("region_model.size_encoder.bias")
    )

    model.region["size_decoder"]["fc1"].weight.data.copy_(
        get_tensor("region_model.size_decoder.fc1.weight")
    )
    model.region["size_decoder"]["fc1"].bias.data.copy_(
        get_tensor("region_model.size_decoder.fc1.bias")
    )
    model.region["size_decoder"]["fc2"].weight.data.copy_(
        get_tensor("region_model.size_decoder.fc2.weight")
    )
    model.region["size_decoder"]["fc2"].bias.data.copy_(
        get_tensor("region_model.size_decoder.fc2.bias")
    )


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
