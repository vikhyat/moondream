import torch
from torch import nn
from PIL import Image
from einops import rearrange
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
import timm


class VisualHolder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual = model

    def forward(self, x):
        return self.visual(x)


class ModelHolder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class LinearPatchEmbedding(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.linear = nn.Linear(588, 1152)
        self.linear.weight.data = conv.weight.data.view(1152, -1)
        if conv.bias is not None:
            self.linear.bias.data = conv.bias.data

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
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
    def __init__(self) -> None:
        super().__init__()

        self.encoder = ModelHolder(
            VisualHolder(timm.create_model("vit_so400m_patch14_siglip_384"))
        )
        self.encoder.model.visual.patch_embed = LinearPatchEmbedding(
            self.encoder.model.visual.patch_embed.proj
        )
        self.encoder.model.visual.attn_pool = nn.Identity()

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

    def __call__(self, image: Image) -> torch.Tensor:
        with torch.no_grad():
            x = (
                self.preprocess(image.convert("RGB"))
                .unsqueeze(0)
                .to(self.device, dtype=self.dtype)
            )
            x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)

            x = self.encoder(x)
            x = self.projection(x)

            return x
