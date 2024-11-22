import json
import os
import tarfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, TypedDict, Union

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

from .preprocess import create_patches


@dataclass
class EncodedImage:
    pos: int
    kv_caches: List[np.ndarray]


SamplingSettings = TypedDict(
    "SamplingSettings",
    {"max_tokens": int},
    total=False,
)

CaptionOutput = TypedDict(
    "CaptionOutput", {"caption": Union[str, Generator[str, None, None]]}
)
QueryOutput = TypedDict(
    "QueryOutput", {"answer": Union[str, Generator[str, None, None]]}
)

DEFAULT_MAX_TOKENS = 1024
MIN_SUPPORTED_VERSION = 1
MAX_SUPPORT_VERSION = 1


class Region:
    pass


class VL:
    def __init__(self, model_path: Optional[str], ort_settings: Dict[str, Any] = {}):
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
                "Model format not recognized. You may need to upgrade the moondream"
                " package."
            )

        self.text_decoders = []

        with tarfile.open(model_path, "r:*") as tar:
            for member in tar.getmembers():
                name = member.name.split("/")[-1]

                f = tar.extractfile(member)
                if f is not None:
                    contents = f.read()
                else:
                    continue

                if name == "vision_encoder.onnx":
                    self.vision_encoder = ort.InferenceSession(contents, **ort_settings)
                elif name == "vision_projection.onnx":
                    self.vision_projection = ort.InferenceSession(
                        contents, **ort_settings
                    )
                elif name == "text_encoder.onnx":
                    self.text_encoder = ort.InferenceSession(contents, **ort_settings)
                elif "text_decoder" in name and name.endswith(".onnx"):
                    self.text_decoders.append(
                        ort.InferenceSession(contents, **ort_settings)
                    )
                elif name == "tokenizer.json":
                    self.tokenizer = Tokenizer.from_buffer(contents)
                elif name == "initial_kv_caches.npy":
                    self.initial_kv_caches = [x for x in np.load(BytesIO(contents))]
                elif name == "config.json":
                    self.config = json.loads(contents)

        assert self.vision_encoder is not None
        assert self.vision_projection is not None
        assert self.text_encoder is not None
        assert len(self.text_decoders) > 0
        assert self.tokenizer is not None
        assert self.initial_kv_caches is not None
        assert self.config is not None

        if type(self.config) != dict or "model_version" not in self.config:
            raise ValueError("Model format not recognized.")
        if (
            self.config["model_version"] < MIN_SUPPORTED_VERSION
            or self.config["model_version"] > MAX_SUPPORT_VERSION
        ):
            raise ValueError(
                "Model version not supported. You may need to upgrade the moondream"
                " package."
            )

        self.special_tokens = self.config["special_tokens"]
        self.templates = self.config["templates"]

    def encode_image(self, image: Union[Image.Image, EncodedImage]) -> EncodedImage:
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
        if type(image) == EncodedImage:
            return image

        image_patches = create_patches(image)  # type: ignore

        patch_emb = self.vision_encoder.run(None, {"input": image_patches})[0]
        patch_emb = np.concatenate([patch_emb[0], patch_emb[1]], axis=-1)
        patch_emb = np.expand_dims(patch_emb, axis=0)
        (inputs_embeds,) = self.vision_projection.run(None, {"input": patch_emb})

        kv_caches = self.initial_kv_caches
        pos = inputs_embeds.shape[-2] + kv_caches[0].shape[-2]

        for i, decoder in enumerate(self.text_decoders):
            inputs_embeds, kv_cache_update = decoder.run(
                None,
                {
                    "inputs_embeds": inputs_embeds,
                    "kv_cache": kv_caches[i],
                },
            )
            kv_caches[i] = np.concatenate([kv_caches[i], kv_cache_update], axis=-2)
        return EncodedImage(pos=pos, kv_caches=kv_caches)

    def _generate(
        self, hidden: np.ndarray, encoded_image: EncodedImage, max_tokens: int
    ) -> Generator[str, None, None]:
        kv_caches = {
            i: np.zeros(
                (
                    *self.initial_kv_caches[0].shape[:-2],
                    2048,
                    self.initial_kv_caches[0].shape[-1],
                ),
                dtype=np.float16,
            )
            for i in range(len(self.text_decoders))
        }
        for i, kv_cache in kv_caches.items():
            kv_cache[:, :, :, :, : encoded_image.pos, :] = encoded_image.kv_caches[i]

        pos = encoded_image.pos
        generated_tokens = 0
        while generated_tokens < max_tokens:
            # Track the original T dimension of the input hidden states, so we can
            # bind the kv cache update accordingly. We can't check it just-in-time
            # because the final 'hidden' output is actually the model's logits.
            og_t = hidden.shape[-2]

            for i, decoder in enumerate(self.text_decoders):
                hidden, kv_cache_update = decoder.run(
                    None,
                    {
                        "inputs_embeds": hidden,
                        "kv_cache": kv_caches[i][:, :, :, :, :pos, :],
                    },
                )
                kv_caches[i][:, :, :, :, pos : pos + og_t, :] = kv_cache_update

            next_token = np.argmax(hidden, axis=-1)[0]
            if next_token == self.special_tokens["eos"]:
                break

            yield self.tokenizer.decode([next_token])
            generated_tokens += 1
            pos += og_t
            (hidden,) = self.text_encoder.run(None, {"input_ids": [[next_token]]})

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: str = "normal",
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> CaptionOutput:
        """
        Generate a caption for the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be captioned.
            settings (Optional[SamplingSettings]): Optional settings for the caption generation.
                If not provided, default settings will be used.

        Returns:
            str: The caption for the input image.
        """
        if "caption" not in self.templates:
            raise ValueError("Model does not support captioning.")
        if length not in self.templates["caption"]:
            raise ValueError(f"Model does not support caption length '{length}'.")

        (input_embeds,) = self.text_encoder.run(
            None, {"input_ids": [self.templates["caption"][length]]}
        )
        if settings is None:
            settings = {}
        max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS)

        encoded_image = self.encode_image(image)

        def generator():
            for t in self._generate(input_embeds, encoded_image, max_tokens):
                yield t

        if stream:
            return {"caption": generator()}
        else:
            out = ""
            for t in generator():
                out += t
            return {"caption": out}

    def query(
        self,
        image: Union[Image.Image, EncodedImage],
        question: str,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> QueryOutput:
        """
        Generate an answer to the input question about the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be queried.
            question (str): The question to be answered.

        Returns:
            str: The answer to the input question about the input image.
        """
        if "query" not in self.templates:
            raise ValueError("Model does not support querying.")

        question_toks = (
            self.templates["query"]["prefix"]
            + self.tokenizer.encode(question).ids
            + self.templates["query"]["suffix"]
        )

        (input_embeds,) = self.text_encoder.run(None, {"input_ids": [question_toks]})
        if settings is None:
            settings = {}
        max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS)

        encoded_image = self.encode_image(image)

        def generator():
            for t in self._generate(input_embeds, encoded_image, max_tokens):
                yield t

        if stream:
            return {"answer": generator()}
        else:
            out = ""
            for t in generator():
                out += t
            return {"answer": out}

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
