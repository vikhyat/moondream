import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
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

        self.mlp = MLP(image_embedding_dim, hidden_dim, model_dim)

    @property
    def device(self):
        return self.mlp.fc1.weight.device

    def forward(self, x):
        return self.mlp(x)


class VisionEncoder(nn.Module):

    def __init__(self, use_flash_attn=False):
        super().__init__()

        self.encoder = EncoderWrapper(use_flash_attn)
        self.projection = VisionProjection()

        self.preprocess = Compose(
            [
                Resize(size=(378, 378), interpolation=InterpolationMode.BICUBIC),
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @property
    def device(self):
        return self.projection.mlp.fc1.weight.device

    @property
    def dtype(self):
        return self.projection.mlp.fc1.weight.dtype

    def __call__(self, images) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        with torch.no_grad():
            # Skip preprocess if images are already tensors
            if not isinstance(images[0], torch.Tensor):
                images = [self.preprocess(image.convert("RGB")) for image in images]

            x = torch.stack(images).to(self.device, dtype=self.dtype)
            x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)

            x = self.encoder(x)
            x = self.projection(x)

            return x
