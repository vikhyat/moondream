import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from PIL import Image

from .layers import attn, layer_norm, linear, mlp
from .weights import VisionModel
from .image_crops import overlap_crop_image, reconstruct_from_crops

if torch.backends.mps.is_available():
    # Non-divisible input sizes are not implemented on MPS device yet.
    # https://github.com/pytorch/pytorch/issues/96056
    def adaptive_avg_pool2d(input, output_size):
        return F.adaptive_avg_pool2d(input.to("cpu"), output_size).to("mps")

else:
    adaptive_avg_pool2d = F.adaptive_avg_pool2d


def encode_image(image: Image.Image, weights: VisionModel) -> torch.Tensor:
    np_image = np.array(image.convert("RGB"))
    crops = overlap_crop_image(np_image, max_crops=12, overlap_margin=4)
    all_crops = np.stack([crops["global_crop"], *crops["local_crops"]], axis=0)
    all_crops = np.transpose(all_crops, (0, 3, 1, 2))
    all_crops = (
        torch.from_numpy(all_crops)
        .to(device=weights.pos_emb.device, dtype=torch.float16)
        .div_(255.0)
        .sub_(0.5)
        .div_(0.5)
    )

    outputs = vision_encoder(all_crops, weights)

    global_features = outputs[0]
    local_features = outputs[1:].view(-1, 27, 27, 1152)

    reconstructed = reconstruct_from_crops(
        local_features,
        crops["tiling"],
        patch_size=1,
        overlap_margin=4,
    )

    reconstructed = reconstructed.permute(2, 0, 1)
    reconstructed = adaptive_avg_pool2d(reconstructed, output_size=(27, 27))
    reconstructed = reconstructed.permute(1, 2, 0).view(729, 1152)
    final_features = torch.cat([global_features, reconstructed], dim=-1)

    return mlp(final_features, weights.proj_mlp)


def vision_encoder(input_BCHW: torch.Tensor, w: VisionModel):
    x = rearrange(
        input_BCHW,
        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
        p1=w.patch_size,
        p2=w.patch_size,
    )  # B3HW -> B(HxW)(3xP1xP2), aka BTC

    x = linear(x, w.patch_emb)
    x = x + w.pos_emb
    for block in w.blocks:
        x = x + attn(layer_norm(x, block.ln1), block.attn)
        x = x + mlp(layer_norm(x, block.ln2), block.mlp)
    x = layer_norm(x, w.post_ln)

    return x
