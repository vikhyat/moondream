from typing import Literal, Optional, Union

import torch
from PIL import Image

from .torch.moondream import MoondreamConfig, MoondreamModel
from .torch.weights import load_weights_into_model
from .types import (
    VLM,
    Base64EncodedImage,
    CaptionOutput,
    DetectOutput,
    EncodedImage,
    PointOutput,
    QueryOutput,
    SamplingSettings,
)
from .version import __version__


class TorchVL(VLM):
    def __init__(
        self,
        *,
        model: str,
    ):
        config = MoondreamConfig()
        self.model = MoondreamModel(config)
        load_weights_into_model(model, self.model)
        self.model.eval()
        # Move model to the appropriate device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)

    def encode_image(
        self, image: Union[Image.Image, EncodedImage]
    ) -> Base64EncodedImage:
        if isinstance(image, EncodedImage):
            assert type(image) == Base64EncodedImage
            return image

        if not self.model:
            raise ValueError("No local model loaded")

        return self.model.encode_image(image)

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: Literal["normal", "short"] = "normal",
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> CaptionOutput:
        if not self.model:
            raise ValueError("No local model loaded")

        encoded_image = (
            self.model.encode_image(image) if isinstance(image, Image.Image) else image
        )
        return self.model.caption(
            encoded_image, length=length, stream=stream, settings=settings
        )

    def query(
        self,
        image: Union[Image.Image, EncodedImage],
        question: str,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> QueryOutput:
        if not self.model:
            raise ValueError("No local model loaded")

        encoded_image = (
            self.model.encode_image(image) if isinstance(image, Image.Image) else image
        )
        return self.model.query(
            encoded_image, question, stream=stream, settings=settings
        )

    def detect(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
    ) -> DetectOutput:
        if not self.model:
            raise ValueError("No local model loaded")

        encoded_image = (
            self.model.encode_image(image) if isinstance(image, Image.Image) else image
        )
        return self.model.detect(encoded_image, object)

    def point(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
    ) -> PointOutput:
        if not self.model:
            raise ValueError("No local model loaded")

        encoded_image = (
            self.model.encode_image(image) if isinstance(image, Image.Image) else image
        )
        return self.model.point(encoded_image, object)
