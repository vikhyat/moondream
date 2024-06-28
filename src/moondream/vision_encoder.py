from typing import Union

import PIL.Image
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import PIL
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
from transformers.utils import is_flash_attn_2_available

try:
    if is_flash_attn_2_available():
        from flash_attn.modules.mha import FlashSelfAttention
    else:
        FlashSelfAttention = None
except ImportError:
    FlashSelfAttention = None


class Attention(nn.Module):

    def __init__(self, dim, num_heads=16, use_flash_attn=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        if use_flash_attn and FlashSelfAttention is not None:
            self.flash_attn = FlashSelfAttention()
        else:
            self.flash_attn = None

        torch.nn.init.kaiming_normal_(
            self.qkv.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_normal_(
            self.proj.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flash_attn is not None:
            qkv = self.qkv(x)
            qkv = rearrange(
                qkv, "... (three h d) -> ... three h d", three=3, h=self.num_heads
            )
            attn_output = self.flash_attn(qkv)
            output = rearrange(attn_output, "... h d -> ... (h d)")
            output = self.proj(output)
            return output
        else:
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)

            x = F.scaled_dot_product_attention(q, k, v)

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            return x


class VitBlock(nn.Module):

    def __init__(self, embed_dim, use_flash_attn=False):
        super().__init__()
        self.attn = Attention(embed_dim, use_flash_attn=use_flash_attn)
        self.mlp = MLP(embed_dim, 4304)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(self, use_flash_attn=False):
        super().__init__()

        embed_len = 729
        embed_dim = 1152

        self.patch_embed = LinearPatchEmbedding()
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.blocks = nn.Sequential(
            *[VitBlock(embed_dim, use_flash_attn=use_flash_attn) for _ in range(27)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class EncoderWrapper(nn.Module):

    def __init__(self, use_flash_attn=False):
        super().__init__()
        self.model = nn.ModuleDict({"visual": VisionTransformer(use_flash_attn)})

    def forward(self, x):
        return self.model["visual"](x)


class LinearPatchEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(588, 1152)

    def forward(self, x):
        b, c, hp1, wp2 = x.shape
        p1, p2 = 14, 14
        h, w = hp1 // p1, wp2 // p2
        x = x.reshape(b, c, h, p1, w, p2)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(b, h * w, c * p1 * p2)

        return self.linear(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

        torch.nn.init.kaiming_normal_(
            self.fc1.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_normal_(
            self.fc2.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class VisionProjection(nn.Module):
    def __init__(self):
        super().__init__()

        image_embedding_dim = 1152
        model_dim = 2048
        hidden_dim = model_dim * 4

        self.mlp = MLP(image_embedding_dim * 2, hidden_dim, model_dim)

    @property
    def device(self):
        return self.mlp.fc1.weight.device

    def forward(self, x):
        return self.mlp(x)


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


class VisionEncoder(nn.Module):

    def __init__(self, use_flash_attn=False):
        super().__init__()

        self.encoder = EncoderWrapper(use_flash_attn)
        self.projection = VisionProjection()
        self.supported_sizes = [(378, 378), (378, 756), (756, 378), (756, 756)]

    @property
    def device(self):
        return self.projection.mlp.fc1.weight.device

    @property
    def dtype(self):
        return self.projection.mlp.fc1.weight.dtype

    def preprocess(self, image: PIL.Image.Image):
        width, height = image.size
        max_dim = max(width, height)
        if max_dim < 512:
            im_size = (378, 378)
        else:
            aspect_ratio = width / height
            im_size = min(
                self.supported_sizes,
                key=lambda size: (
                    abs((size[1] / size[0]) - aspect_ratio),
                    abs(size[0] - width) + abs(size[1] - height),
                ),
            )

        return Compose(
            [
                Resize(size=im_size, interpolation=InterpolationMode.BICUBIC),
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )(image)

    def forward(
        self, images: Union[PIL.Image.Image, list[PIL.Image.Image], torch.Tensor]
    ) -> torch.Tensor:
        im_list = None
        if isinstance(images, torch.Tensor):
            # Input must have dimensions (B, C, H, W)
            assert (
                len(images.shape) == 4
            ), "Tensor input must have dimensions (B, C, H, W)"
            im_list = list(images)
        elif isinstance(images, PIL.Image.Image):
            im_list = [images]
        elif isinstance(images, list):
            im_list = images
        else:
            raise ValueError(
                "Input must be a PIL image, list of PIL images, or a tensor"
            )

        # Preprocess unless the images are already tensors (indicating that
        # they have already been preprocessed)
        if not isinstance(im_list[0], torch.Tensor):
            im_list = [self.preprocess(im.convert("RGB")) for im in im_list]

        patches = [create_patches(im) for im in im_list]
        flat_patches = [patch for image_patches in patches for patch in image_patches]

        # Images may be variable size, and need to be resized to a common size after
        # creating patches.
        resized_images = [
            F.interpolate(im.unsqueeze(0), size=(378, 378), mode="bilinear")
            for im in im_list
        ]

        combined_images = torch.cat([*resized_images, *flat_patches], dim=0)
        combined_images = combined_images.to(self.device, dtype=self.dtype)

        combined_features = self.encoder(combined_images)

        full_img_features = combined_features[: len(im_list)]
        patch_features = (
            combined_features[len(im_list) :].transpose(1, 2).view(-1, 1152, 27, 27)
        )

        # Reshape patch features back to their original structure
        reshaped_patch_features = []
        patch_idx = 0
        for i, patch_set in enumerate(patches):
            if len(patch_set) == 0:
                reshaped_patch_features.append(
                    full_img_features[i].transpose(0, 1).view(1152, 27, 27)
                )
            else:
                sample_features = []
                for row_patches in patch_set:
                    row_len = len(row_patches)
                    row_features = patch_features[
                        patch_idx : patch_idx + row_len
                    ]  # row_len, T, C
                    row_features = torch.cat(
                        list(row_features), dim=2
                    )  # T, C * row_len
                    patch_idx += row_len
                    sample_features.append(row_features)
                sample_features = torch.cat(sample_features, dim=1)
                sample_features = F.interpolate(
                    sample_features.unsqueeze(0), size=(27, 27), mode="bilinear"
                ).squeeze(0)
                reshaped_patch_features.append(sample_features)
        reshaped_patch_features = (
            torch.stack(reshaped_patch_features).view(-1, 1152, 729).transpose(1, 2)
        )

        final_features = torch.cat([full_img_features, reshaped_patch_features], dim=2)

        return self.projection(final_features)
