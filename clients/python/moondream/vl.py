import json
import os
import tarfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, TypedDict, Union
from moondream.utils import BASE_URL, validate_image

import json
from pathlib import Path
import uuid
from PIL import Image
import httpx

import numpy as np
import onnxruntime as ort
from PIL import Image
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

CaptionOutput = TypedDict(
    "CaptionOutput", {"caption": Union[str, Generator[str, None, None]]}
)
QueryOutput = TypedDict(
    "QueryOutput", {"answer": Union[str, Generator[str, None, None]]}
)

DEFAULT_MAX_TOKENS = 1024
LATEST_SUPPORTED_VERSION = 0

# A Hotdog is a


class Region:
    pass


class VL:
    def __init__(
        self,
        model_path: Optional[str] = None,
        ort_settings: Dict[str, Any] = {},
        api_key: Optional[str] = None,
        model_endpoint: Optional[str] = None,
    ):
        """
        Initialize the Moondream VL (Vision Language) model.

        Args:
            model_path (Optional[str]): Path to the model file for local inference. Required if running locally.
            ort_settings (Dict[str, Any]): ONNX Runtime settings for local inference configuration. Defaults to {}.
            api_key (Optional[str]): API key from the Moondream console for cloud inference. Required if running in cloud mode. Sign up at https://moondream.ai
            model_endpoint (Optional[str]): Endpoint of your fine-tuned model hosted on Moondream cloud. If not provided, uses the base Moondream model.

        Raises:
            ValueError: If neither model_path nor api_key is provided, or if both are provided.

        Note:
            You must provide either model_path (for local inference) or api_key (for cloud inference), but not both.
        """
        if model_path is not None and api_key is not None:
            raise ValueError(
                "Cannot provide both model_path and api_key. Use model_path for local inference OR api_key for cloud inference."
            )
        if model_path is None and api_key is None:
            raise ValueError(
                "Must provide either model_path (for local inference) or api_key (for cloud inference)."
            )

        self.inference_type = "local" if model_path else "cloud"

        if model_path is not None:
            if not os.path.isfile(model_path):
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
                        self.vision_encoder = ort.InferenceSession(
                            contents, **ort_settings
                        )
                    elif name == "vision_projection.onnx":
                        self.vision_projection = ort.InferenceSession(
                            contents, **ort_settings
                        )
                    elif name == "text_encoder.onnx":
                        self.text_encoder = ort.InferenceSession(
                            contents, **ort_settings
                        )
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
            if self.config["model_version"] > LATEST_SUPPORTED_VERSION:
                raise ValueError(
                    "Model version not supported. You may need to upgrade the moondream"
                    " package."
                )

            self.special_tokens = self.config["special_tokens"]
            self.templates = self.config["templates"]
        else:
            self.api_key = api_key
            self.model_endpoint = model_endpoint
            self.httpx_client = httpx.Client(timeout=20.0)

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
        if self.inference_type != "local":
            raise ValueError(
                "Vision Language model is not configured for local inference."
            )

        if type(image) == EncodedImage:
            return image

        image_patches = create_patches(image)  # type: ignore

        patch_emb = self.vision_encoder.run(None, {"input": image_patches})[0]
        patch_emb = np.concatenate([patch_emb[0], patch_emb[1]], axis=-1)
        patch_emb = np.expand_dims(patch_emb, axis=0)
        (inputs_embeds,) = self.vision_projection.run(None, {"input": patch_emb})

        kv_caches = self.initial_kv_caches
        for i, decoder in enumerate(self.text_decoders):
            inputs_embeds, kv_caches[i] = decoder.run(
                None,
                {
                    "inputs_embeds": inputs_embeds,
                    "kv_cache": kv_caches[i],
                },
            )
        return EncodedImage(kv_caches=kv_caches)

    def _generate(
        self, hidden: np.ndarray, encoded_image: EncodedImage, max_tokens: int
    ) -> Generator[str, None, None]:
        kv_caches = {
            i: encoded_image.kv_caches[i] for i in range(len(self.text_decoders))
        }

        generated_tokens = 0
        while generated_tokens < max_tokens:
            for i, decoder in enumerate(self.text_decoders):
                hidden, kv_caches[i] = decoder.run(
                    None, {"inputs_embeds": hidden, "kv_cache": kv_caches[i]}
                )

            next_token = np.argmax(hidden, axis=-1)[0]
            if next_token == self.special_tokens["eos"]:
                break

            yield self.tokenizer.decode([next_token])
            generated_tokens += 1
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
        This supports both local and cloud inference.

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

    def _make_query_request(
        self,
        image_buffer: BytesIO,
        prompt: str,
    ) -> dict:

        request_id = str(uuid.uuid4())

        request_files = {"content": (f"{request_id}.jpg", image_buffer, "image/jpeg")}

        request_headers = {
            "X-MD-Auth": self.api_key,
        }
        request_data = {"body": json.dumps({"prompt": prompt})}

        request_url = f"{BASE_URL}/default"
        if self.model_endpoint:
            request_url = f"{BASE_URL}/{self.model_endpoint}"

        response = self.httpx_client.post(
            request_url,
            files=request_files,
            data=request_data,  # Send as form data
            headers=request_headers,
        )
        response.raise_for_status()

        return dict(response.json())

    def query(
        self,
        image: Union[Image.Image, EncodedImage, Path, str],
        question: str,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> QueryOutput:
        """
        Generate an answer to the input question about the input image.

        Args:
            image (Union[Image.Image, EncodedImage, Path, str]): The input image to be queried.
                Can be a PIL Image, encoded image, file path, or URL string.
            question (str): The question to be answered about the image.
            stream (bool, optional): If True, returns a generator that yields tokens as they're generated.
                Defaults to False.
            settings (Optional[SamplingSettings], optional): Generation settings like max_tokens.
                Defaults to None.

        Returns:
            QueryOutput: A dictionary containing the answer under the 'answer' key.
                If stream=True, the answer will be a generator yielding tokens.
                If stream=False, the answer will be a complete string.
        """

        if self.inference_type == "cloud":
            if type(image) == EncodedImage:
                raise ValueError("Cloud inference does not support encoded images.")

            image_to_query = validate_image(image)

            server_response = self._make_query_request(image_to_query, question)
            return {"answer": server_response["result"]}

        elif self.inference_type == "local":
            if "query" not in self.templates:
                raise ValueError("Model does not support querying.")
            if type(image) != EncodedImage:
                raise ValueError(
                    "Local inference does not support unencoded images, encode the image using `encode_image` first."
                )

            question_toks = (
                self.templates["query"]["prefix"]
                + self.tokenizer.encode(question).ids
                + self.templates["query"]["suffix"]
            )

            (input_embeds,) = self.text_encoder.run(
                None, {"input_ids": [question_toks]}
            )
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
