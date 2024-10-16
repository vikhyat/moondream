import torch
import math
from typing import List, Tuple, Union
from einops import rearrange
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2.functional import (
    resize as tv_resize,
    to_image,
    to_dtype,
    normalize,
)

from .weights import VisionModel, load_from_safetensors
from .layers import attn, linear, layer_norm, mlp


def im_resize(
    image: Image.Image,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
) -> Image.Image:
    """
    The 'resize' function from torchvision has bad type signatures.
    it accepts both PIL images and torch tensors,  but the type signature
    only allows tensors.
    """
    return tv_resize(
        image,  # type: ignore
        size,
        InterpolationMode.BICUBIC,
    )


def create_patches(
    image: Image.Image, image_patch_size=378
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    Split the given image into a variable number of patches depending upon its
    resolution.
    """
    # Start off with the global patch.
    patches = [im_resize(image, [image_patch_size, image_patch_size])]

    # Find the closest resolution template.
    res_templates = [(1, 2), (2, 1), (2, 2)]
    im_width, im_height = image.size
    max_dim = max(im_width, im_height)
    if max_dim < image_patch_size * 1.4:
        # If the image is already small, we just do a single patch that is a
        # duplicate of the global patch. This creates a small amount of
        # redundant computation now, but it is simpler and future-proofs us
        # if/when we condition the vision encoder on the patch type.
        res_template = (1, 1)
        patches.append(patches[0])
    else:
        aspect_ratio = im_width / im_height
        res_template = min(
            res_templates, key=lambda size: abs((size[1] / size[0]) - aspect_ratio)
        )
        # TODO: Actually implement patching... just going to put in the global
        # patch for now to make progress on other aspects.
        patches.append(patches[0])

    return patches, res_template


def encode_image(image: Image.Image, weights: VisionModel) -> torch.Tensor:
    patches, res_template = create_patches(image.convert("RGB"))
    patches = torch.stack(
        [
            normalize(
                to_dtype(to_image(patch), torch.float16, scale=True),
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
            for patch in patches
        ]
    )

    outputs = vision_encoder(patches, weights)

    # TODO: Merge sub-image patch outputs properly... for now we'll just assume
    # that the global patch is repeated.
    assert outputs.shape[0] == 2, "Expected single image patch."
    outputs = torch.cat([outputs[0], outputs[1]], dim=-1)

    return mlp(outputs, weights.proj_mlp)


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
