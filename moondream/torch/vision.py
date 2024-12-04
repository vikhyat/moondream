from typing import List, Tuple

import torch
from einops import rearrange
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2.functional import normalize
from torchvision.transforms.v2.functional import resize as tv_resize
from torchvision.transforms.v2.functional import (
    to_dtype,
    to_image,
    normalize,
    pil_to_tensor,
)

from .layers import attn, layer_norm, linear, mlp
from .weights import VisionModel, load_from_safetensors
import torch
from typing import Tuple, List


def im_resize(
    image: Image.Image,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
) -> Image.Image:
    """
    Resize a PIL image using torchvision's resize function.
    """
    return tv_resize(
        image,  # type: ignore
        size,
        interpolation,
    )


def create_patches(image, patch_size=(378, 378)):
    assert image.dim() == 3, "Image must be in CHW format"

    _, height, width = image.shape  # Channels, Height, Width
    patch_height, patch_width = patch_size

    if height == patch_height and width == patch_width:
        return []

    # Iterate over the image and create patches
    patches = []
    for i in range(0, height, patch_height):
        row_patches = []
        for j in range(0, width, patch_width):
            patch = image[:, i : i + patch_height, j : j + patch_width]
            row_patches.append(patch)
        patches.append(torch.stack(row_patches))
    return patches


def preprocess(image: Image.Image) -> torch.Tensor:
    supported_sizes = [(378, 378), (378, 756), (756, 378), (756, 756)]
    width, height = image.size
    max_dim = max(width, height)

    if max_dim < 512:
        im_size = (378, 378)
    else:
        aspect_ratio = width / height
        im_size = min(
            supported_sizes,
            key=lambda size: (
                abs((size[1] / size[0]) - aspect_ratio),
                abs(size[0] - width) + abs(size[1] - height),
            ),
        )

    # Resize the image while it's still a PIL Image
    image = im_resize(image, im_size)

    # Convert the PIL Image to a tensor and scale pixel values to [0, 1]
    tensor = pil_to_tensor(image).float().div(255)

    # Apply same normalization
    tensor = normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Convert tensor to desired dtype
    tensor = tensor.to(dtype=torch.float16)

    return tensor


def save_tensor(tensor, filepath):
    """Save a tensor to a file with error handling"""
    try:
        torch.save(tensor, filepath)
        return True
    except Exception as e:
        print(f"Error saving tensor: {e}")
        return False


def encode_image(image: Image.Image, weights: VisionModel) -> torch.Tensor:
    image = preprocess(image.convert("RGB"))
    patches = create_patches(image)

    flat_patches = [patch for row_patches in patches for patch in row_patches]

    # Resize the image to common size
    resized_image = F.interpolate(image.unsqueeze(0), size=(378, 378), mode="bilinear")
    resized_image = resized_image.squeeze(0)  # Remove batch dimension

    # Combine image with patches
    if flat_patches:
        combined_images = torch.stack(
            [resized_image] + flat_patches
        )  # Use stack instead of cat
        # print(combined_images.shape) #(torch.Size([3, 3, 378, 378]))
    else:
        print("no flat patches")
        combined_images = resized_image.unsqueeze(0)  # Add batch dimension
        # print(combined_images.shape) #torch.Size([1, 3, 378, 378])

    combined_features = vision_encoder(combined_images, weights)
    # save_tensor(combined_features, '/workspace/moondream_private/moondream/torch/torch-a-img.pt')

    # Split features for full image and patches
    full_img_features = combined_features[0:1]  # Keep batch dimension
    patch_features = combined_features[1:].transpose(1, 2).view(-1, 1152, 27, 27)

    # Handle patch features
    if len(patches) == 0:
        reshaped_patch_features = []

        reshaped_patch_features.append(
            full_img_features[0].transpose(0, 1).view(1152, 27, 27)
        )
        reshaped_patch_features = (
            torch.stack(reshaped_patch_features).view(-1, 1152, 729).transpose(1, 2)
        )

    else:
        sample_features = []
        patch_idx = 0
        for row_patches in patches:
            row_len = len(row_patches)
            row_features = patch_features[
                patch_idx : patch_idx + row_len
            ]  # row_len, T, C
            row_features = torch.cat(list(row_features), dim=2)  # T, C * row_len
            patch_idx += row_len
            sample_features.append(row_features)

        sample_features = torch.cat(sample_features, dim=1)
        reshaped_patch_features = F.adaptive_avg_pool2d(
            sample_features, output_size=(27, 27)
        ).unsqueeze(
            0
        )  # Add batch dimension back

        # Reshape and combine features
        reshaped_patch_features = reshaped_patch_features.reshape(
            -1, 1152, 729
        ).transpose(1, 2)

    final_features = torch.cat([full_img_features, reshaped_patch_features], dim=2)

    return mlp(final_features, weights.proj_mlp)[0] # uniformly remove batch dim


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
