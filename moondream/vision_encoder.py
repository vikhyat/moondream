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


class VisionEncoder(nn.Module):
    def __init__(self, model_path: str = "model") -> None:
        super().__init__()

        self.model = torch.jit.load(f"{model_path}/vision.pt")
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
        return self.model.projection.mlp1.fc1.weight.device

    @property
    def dtype(self):
        return self.model.projection.mlp1.fc1.weight.dtype

    def __call__(self, image: Image) -> torch.Tensor:
        with torch.no_grad():
            image_vec = (
                self.preprocess(image.convert("RGB"))
                .unsqueeze(0)
                .to(self.device, dtype=self.dtype)
            )
            image_vec = rearrange(
                image_vec, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14
            )
            return self.model(image_vec)
