import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union, Tuple
from PIL import Image

from .layers import attn, layer_norm, linear, mlp
from .image_crops import overlap_crop_image
from .config import VisionConfig

if torch.backends.mps.is_available():
    # Non-divisible input sizes are not implemented on MPS device yet.
    # https://github.com/pytorch/pytorch/issues/96056
    def adaptive_avg_pool2d(input, output_size):
        return F.adaptive_avg_pool2d(input.to("cpu"), output_size).to("mps")

else:
    adaptive_avg_pool2d = F.adaptive_avg_pool2d

DeviceLike = Union[str, torch.device, int]


def prepare_crops(
    image: Image.Image, config: VisionConfig, device: DeviceLike
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


def create_patches(x, patch_size):
    # Original shape: [B, C, H, W]
    B, C, H, W = x.shape
    P1 = P2 = patch_size

    # Step 1: Split H and W dimensions into patches
    # [B, C, H/P1, P1, W/P2, P2]
    x = x.reshape(B, C, H // P1, P1, W // P2, P2)

    # Step 2: Rearrange dimensions to match target shape
    # [B, H/P1, W/P2, C, P1, P2]
    x = x.permute(0, 2, 4, 1, 3, 5)

    # Step 3: Combine dimensions to get final shape
    # [B, (H/P1)*(W/P2), C*P1*P2]
    x = x.reshape(B, (H // P1) * (W // P2), C * P1 * P2)

    return x


def vision_encoder(input_BCHW: torch.Tensor, w: nn.Module, config: VisionConfig):
    x = create_patches(input_BCHW, config.enc_patch_size)

    x = linear(x, w.patch_emb)
    x = x + w.pos_emb
    for block in w.blocks:
        x = x + attn(layer_norm(x, block.ln1), block.attn, n_heads=config.enc_n_heads)
        x = x + mlp(layer_norm(x, block.ln2), block.mlp)
    x = layer_norm(x, w.post_ln)

    return x


def vision_projection(
    global_features: torch.Tensor,
    reconstructed: torch.Tensor,
    w: nn.Module,
    config: VisionConfig,
):
    reconstructed = reconstructed.permute(2, 0, 1)
    reconstructed = adaptive_avg_pool2d(
        reconstructed, output_size=(config.enc_n_layers, config.enc_n_layers)
    )
    reconstructed = reconstructed.permute(1, 2, 0).view(729, config.enc_dim)
    final_features = torch.cat([global_features, reconstructed], dim=-1)
    return mlp(final_features, w.proj_mlp)


def build_vision_model(config: VisionConfig, dtype: torch.dtype):
    patch_dim = config.enc_patch_size * config.enc_patch_size * config.in_channels
    grid_size = config.crop_size // config.enc_patch_size
    num_patches = grid_size * grid_size

    vision = nn.ModuleDict(
        {
            "patch_emb": nn.Linear(patch_dim, config.enc_dim, dtype=dtype),
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln1": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": nn.Linear(
                                        config.enc_dim, 3 * config.enc_dim, dtype=dtype
                                    ),
                                    "proj": nn.Linear(
                                        config.enc_dim, config.enc_dim, dtype=dtype
                                    ),
                                }
                            ),
                            "ln2": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "mlp": nn.ModuleDict(
                                {
                                    "fc1": nn.Linear(
                                        config.enc_dim, config.enc_ff_dim, dtype=dtype
                                    ),
                                    "fc2": nn.Linear(
                                        config.enc_ff_dim, config.enc_dim, dtype=dtype
                                    ),
                                }
                            ),
                        }
                    )
                    for _ in range(config.enc_n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.enc_dim, dtype=dtype),
            "proj_mlp": nn.ModuleDict(
                {
                    "fc1": nn.Linear(
                        config.enc_dim * 2, config.proj_inner_dim, dtype=dtype
                    ),
                    "fc2": nn.Linear(
                        config.proj_inner_dim, config.proj_out_dim, dtype=dtype
                    ),
                }
            ),
        }
    )
    vision.pos_emb = nn.Parameter(
        torch.zeros(1, num_patches, config.enc_dim, dtype=dtype)
    )
    return vision
