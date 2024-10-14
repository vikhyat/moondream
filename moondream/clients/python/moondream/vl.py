import onnx
import onnxruntime as ort
import numpy as np
import os
import tarfile

from typing import Generator, List, Union, Optional, Dict, TypedDict
from PIL import Image
from dataclasses import dataclass
from tokenizers import Tokenizer

from .preprocess import create_patches


@dataclass
class EncodedImage:
    kv_caches: List[np.ndarray]


SamplingSettings = TypedDict(
    "SamplingSettings",
    {"max_tokens": int},
    total=False,
)

DEFAULT_MAX_TOKENS = 1024


class Region:
    pass


class VL:
    def __init__(self, model_path: Optional[str]):
        """
        Initialize the Moondream VL (Vision Language) model.

        Args:
            model_path (str): The path to the model file.

        Returns:
            None
        """

        if model_path is None or not os.path.isfile(model_path):
            raise ValueError("Model path is invalid or file does not exist.")

        if not tarfile.is_tarfile(model_path):
            raise ValueError(
                "Model format not recognized. You may need to upgrade the moondream package."
            )

        self.text_decoders = []

        with tarfile.open(model_path, "r:*") as tar:
            for member in tar.getmembers():
                name = member.name

                f = tar.extractfile(member)
                if f is not None:
                    contents = f.read()
                else:
                    continue

                if name == "vision_encoder.onnx":
                    self.vision_encoder = ort.InferenceSession(contents)
                elif name == "vision_projection.onnx":
                    self.vision_projection = ort.InferenceSession(contents)
                elif name == "text_encoder.onnx":
                    self.text_encoder = ort.InferenceSession(contents)
                elif name == "tokenizer.json":
                    self.tokenizer = Tokenizer.from_buffer(contents)
                elif "text_decoder" in name and name.endswith(".onnx"):
                    self.text_decoders.append(ort.InferenceSession(contents))

        assert self.vision_encoder is not None
        assert self.vision_projection is not None
        assert self.tokenizer is not None
        assert self.text_encoder is not None
        assert len(self.text_decoders) > 0

        self.eos_token_id = self.tokenizer.encode("<|endoftext|>").ids[0]
        self.caption_prefix = self.tokenizer.encode("\n\nCaption:").ids

    def encode_image(self, image: Image.Image) -> EncodedImage:
        """
        Preprocess the image by running it through the model.

        This method is useful if the user wants to make multiple queries with the same image.
        The output is not guaranteed to be backward-compatible across version updates,
        and should not be persisted out of band.

        Args:
            image (Image.Image): The input image to be encoded.

        Returns:
            The encoded representation of the image.
        """
        (inputs_embeds_0,) = self.text_encoder.run(
            None, {"input_ids": np.array([[self.eos_token_id]])}
        )
        image_patches = create_patches(image)
        patch_emb = self.vision_encoder.run(None, {"input": image_patches})[0]
        patch_emb = np.concatenate([patch_emb[0], patch_emb[1]], axis=-1)
        patch_emb = np.expand_dims(patch_emb, axis=0)
        (inputs_embeds_1,) = self.vision_projection.run(None, {"input": patch_emb})
        inputs_embeds = np.concatenate([inputs_embeds_0, inputs_embeds_1], axis=1)

        kv_caches = [
            np.ndarray(
                [24 // len(self.text_decoders), 2, 1, 32, 0, 64], dtype=np.float16
            )
            for _ in self.text_decoders
        ]
        for i, decoder in enumerate(self.text_decoders):
            inputs_embeds, kv_caches[i] = decoder.run(
                None, {"inputs_embeds": inputs_embeds, "kv_cache": kv_caches[i]}
            )

        return EncodedImage(kv_caches=kv_caches)

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        settings: Optional[SamplingSettings] = None,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a caption for the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be captioned.
            settings (Optional[SamplingSettings]): Optional settings for the caption generation.
                If not provided, default settings will be used.

        Returns:
            str: The caption for the input image.
        """
        if type(self.caption_prefix) == list:
            (self.caption_prefix,) = self.text_encoder.run(
                None, {"input_ids": [self.caption_prefix]}
            )

        if settings is None:
            settings = {}
        max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS)

        if type(image) != EncodedImage:
            image = self.encode_image(image)  # type: ignore

        hidden = self.caption_prefix
        kv_caches = {i: image.kv_caches[i] for i in range(len(self.text_decoders))}
        generated_tokens = 0
        while generated_tokens < max_tokens:
            for i, decoder in enumerate(self.text_decoders):
                hidden, kv_caches[i] = decoder.run(
                    None, {"inputs_embeds": hidden, "kv_cache": kv_caches[i]}
                )

            next_token = np.argmax(hidden, axis=-1)[0]
            if next_token == self.eos_token_id:
                break

            yield self.tokenizer.decode([next_token])
            generated_tokens += 1
            (hidden,) = self.text_encoder.run(None, {"input_ids": [[next_token]]})

    def query(self, image: Union[Image.Image, EncodedImage], question: str) -> str:
        """
        Generate an answer to the input question about the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be queried.
            question (str): The question to be answered.

        Returns:
            str: The answer to the input question about the input image.
        """
        return "To be implemented."

    def detect(
        self, image: Union[Image.Image, EncodedImage], object: str
    ) -> List[Region]:
        """
        Detect and localize the specified object in the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be analyzed.
            object (str): The object to be detected in the image.

        Returns:
            List[Region]: A list of Region objects representing the detected instances of the specified object.
        """
        return []
