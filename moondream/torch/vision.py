import torch
import torch.nn.functional as F
import numpy as np

from typing import Union, Tuple
from einops import rearrange
from PIL import Image

from .layers import attn, layer_norm, linear, mlp
from .weights import VisionModel
from .image_crops import overlap_crop_image, reconstruct_from_crops
from .config import VisionConfig

if torch.backends.mps.is_available():
    # Non-divisible input sizes are not implemented on MPS device yet.
    # https://github.com/pytorch/pytorch/issues/96056
    def adaptive_avg_pool2d(input, output_size):
        return F.adaptive_avg_pool2d(input.to("cpu"), output_size).to("mps")

else:
    adaptive_avg_pool2d = F.adaptive_avg_pool2d


def prepare_crops(
    image: Image.Image, config: VisionConfig, device: Union[str, torch.device, int]
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    np_image = np.array(image.convert("RGB"))
    overlap_crops = overlap_crop_image(
        np_image, max_crops=config.max_crops, overlap_margin=config.overlap_margin
    )
    all_crops = overlap_crops["crops"]
    all_crops = np.transpose(all_crops, (0, 3, 1, 2))
    all_crops = (
        torch.from_numpy(all_crops)
        .to(device=device, dtype=torch.float16)
        .div_(255.0)
        .sub_(0.5)
        .div_(0.5)
    )
    return all_crops, overlap_crops["tiling"]


def vision_encoder(input_BCHW: torch.Tensor, w: VisionModel, config: VisionConfig):
    x = rearrange(
        input_BCHW,
        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
        p1=config.enc_patch_size,
        p2=config.enc_patch_size,
    )  # B3HW -> B(HxW)(3xP1xP2), aka BTC

    x = linear(x, w.patch_emb)
    x = x + w.pos_emb
    for block in w.blocks:
        x = x + attn(layer_norm(x, block.ln1), block.attn, n_heads=config.enc_n_heads)
        x = x + mlp(layer_norm(x, block.ln2), block.mlp)
    x = layer_norm(x, w.post_ln)

    return x


def vision_projection(
    global_features: torch.Tensor, reconstructed: torch.Tensor, w: VisionModel
):
    reconstructed = reconstructed.permute(2, 0, 1)
    reconstructed = adaptive_avg_pool2d(reconstructed, output_size=(27, 27))
    reconstructed = reconstructed.permute(1, 2, 0).view(729, 1152)
    final_features = torch.cat([global_features, reconstructed], dim=-1)
    return mlp(final_features, w.proj_mlp)


def encode_image(
    image: Image.Image, weights: VisionModel, config: VisionConfig
) -> torch.Tensor:
    # This is split into sub-functions to allow sections to be compiled without
    # graph breaks, which is needed if we want to enable reduce-overhead mode.
    # `vision_encoder` and `vision_projection` can be compiled if needed.

    all_crops, tiling = prepare_crops(image, config, device=weights.pos_emb.device)

    outputs = vision_encoder(all_crops, weights, config)

    global_features = outputs[0]
    local_features = outputs[1:].view(-1, 27, 27, 1152)
    reconstructed = reconstruct_from_crops(
        local_features, tiling, patch_size=1, overlap_margin=config.overlap_margin
    )

    return vision_projection(global_features, reconstructed, weights)
