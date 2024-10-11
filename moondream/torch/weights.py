import torch
import math
import safetensors
from typing import Dict, Union, Literal, List, Any
from contextlib import contextmanager
from dataclasses import dataclass

from .layers import LinearWeights, LayerNormWeights, MLPWeights, AttentionWeights


@dataclass
class VisionBlock:
    ln1: LayerNormWeights
    attn: AttentionWeights
    ln2: LayerNormWeights
    mlp: MLPWeights


@dataclass
class VisionModel:
    patch_size: int
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
class MoondreamModel:
    vision: VisionModel
    text: TextModel


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

        yield get_tensor


def load_from_safetensors(
    safetensors_file: str,
    vision_blocks: int = 27,
    text_blocks: int = 24,
) -> MoondreamModel:
    with safetensors_open(safetensors_file) as get_tensor:
        ## Vision encoder
        prefix = "vision_encoder.encoder.model.visual.patch_embed.linear"
        patch_emb = LinearWeights(
            weight=get_tensor(f"{prefix}.weight"), bias=get_tensor(f"{prefix}.bias")
        )
        patch_size = int(math.sqrt(patch_emb.weight.shape[1] // 3))
        pos_emb = get_tensor("vision_encoder.encoder.model.visual.pos_embed")
        post_ln = LayerNormWeights(
            weight=get_tensor("vision_encoder.encoder.model.visual.norm.weight"),
            bias=get_tensor("vision_encoder.encoder.model.visual.norm.bias"),
        )
        blocks = []
        for i in range(vision_blocks):
            prefix = f"vision_encoder.encoder.model.visual.blocks.{i}"
            blocks.append(
                VisionBlock(
                    ln1=LayerNormWeights(
                        weight=get_tensor(f"{prefix}.norm1.weight"),
                        bias=get_tensor(f"{prefix}.norm1.bias"),
                    ),
                    attn=AttentionWeights(
                        qkv=LinearWeights(
                            weight=get_tensor(f"{prefix}.attn.qkv.weight"),
                            bias=get_tensor(f"{prefix}.attn.qkv.bias"),
                        ),
                        proj=LinearWeights(
                            weight=get_tensor(f"{prefix}.attn.proj.weight"),
                            bias=get_tensor(f"{prefix}.attn.proj.bias"),
                        ),
                        n_heads=16,
                    ),
                    ln2=LayerNormWeights(
                        weight=get_tensor(f"{prefix}.norm2.weight"),
                        bias=get_tensor(f"{prefix}.norm2.bias"),
                    ),
                    mlp=MLPWeights(
                        fc1=LinearWeights(
                            weight=get_tensor(f"{prefix}.mlp.fc1.weight"),
                            bias=get_tensor(f"{prefix}.mlp.fc1.bias"),
                        ),
                        fc2=LinearWeights(
                            weight=get_tensor(f"{prefix}.mlp.fc2.weight"),
                            bias=get_tensor(f"{prefix}.mlp.fc2.bias"),
                        ),
                    ),
                )
            )
        proj_mlp = MLPWeights(
            fc1=LinearWeights(
                weight=get_tensor("vision_encoder.projection.mlp.fc1.weight"),
                bias=get_tensor("vision_encoder.projection.mlp.fc1.bias"),
            ),
            fc2=LinearWeights(
                weight=get_tensor("vision_encoder.projection.mlp.fc2.weight"),
                bias=get_tensor("vision_encoder.projection.mlp.fc2.bias"),
            ),
            act="gelu_approx",
        )
        vision = VisionModel(
            patch_size=patch_size,
            patch_emb=patch_emb,
            pos_emb=pos_emb,
            blocks=blocks,
            post_ln=post_ln,
            proj_mlp=proj_mlp,
        )

        ## Text decoder model
        wte = get_tensor("text_model.transformer.embd.wte.weight")
        post_ln = LayerNormWeights(
            weight=get_tensor("text_model.lm_head.ln.weight"),
            bias=get_tensor("text_model.lm_head.ln.bias"),
        )
        lm_head = LinearWeights(
            weight=get_tensor("text_model.lm_head.linear.weight"),
            bias=get_tensor("text_model.lm_head.linear.bias"),
        )
        blocks = []
        for i in range(text_blocks):
            prefix = f"text_model.transformer.h.{i}"
            blocks.append(
                TextBlock(
                    ln=LayerNormWeights(
                        weight=get_tensor(f"{prefix}.ln.weight"),
                        bias=get_tensor(f"{prefix}.ln.bias"),
                    ),
                    attn=AttentionWeights(
                        qkv=LinearWeights(
                            weight=get_tensor(f"{prefix}.mixer.Wqkv.weight"),
                            bias=get_tensor(f"{prefix}.mixer.Wqkv.bias"),
                        ),
                        proj=LinearWeights(
                            weight=get_tensor(f"{prefix}.mixer.out_proj.weight"),
                            bias=get_tensor(f"{prefix}.mixer.out_proj.bias"),
                        ),
                        n_heads=32,
                    ),
                    mlp=MLPWeights(
                        fc1=LinearWeights(
                            weight=get_tensor(f"{prefix}.mlp.fc1.weight"),
                            bias=get_tensor(f"{prefix}.mlp.fc1.bias"),
                        ),
                        fc2=LinearWeights(
                            weight=get_tensor(f"{prefix}.mlp.fc2.weight"),
                            bias=get_tensor(f"{prefix}.mlp.fc2.bias"),
                        ),
                        act="gelu_approx",
                    ),
                )
            )
        text = TextModel(wte=wte, blocks=blocks, post_ln=post_ln, lm_head=lm_head)

    return MoondreamModel(vision=vision, text=text)


if __name__ == "__main__":
    weights = load_from_safetensors("model.safetensors")
    print(weights)
