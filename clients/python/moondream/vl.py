import json
import os
import tarfile
from dataclasses import dataclass
from io import BytesIO
from typing import Generator, List, Optional, TypedDict, Union
import numpy as np
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

Region = TypedDict(
    "Region", {"x_min": float, "y_min": float, "x_max": int, "y_max": float}
)
DetectOutput = TypedDict("DetectOutput", {"objects": List[Region]})

DEFAULT_MAX_TOKENS = 1024
MIN_SUPPORTED_VERSION = 1
MAX_SUPPORT_VERSION = 1


def prepare_kv_caches(initial_kv_caches, encoded_image):
    """
    Initializes the key-value caches required for the transformer inference session.

    Allocates a zero-initialized array to store the kv-cache states and copies over any
    pre-existing cache states from the encoded image.

    Args:
        initial_kv_caches: Base kv-cache arrays to build upon
        encoded_image: Pre-computed kv-cache states from the image encoder
    Returns:
        Dictionary mapping decoder indices to their initialized kv-cache arrays
    """
    kv_caches = {
        i: np.zeros(
            (
                *initial_kv_caches[0].shape[:-2],
                2048,  # max sequence length
                initial_kv_caches[0].shape[-1],
            ),
            dtype=np.float16,
        )
        for i in range(len(initial_kv_caches))
    }
    for i, kv_cache in kv_caches.items():
        kv_cache[:, :, :, :, : encoded_image.pos, :] = encoded_image.kv_caches[i]
    return kv_caches


def run_decoders(text_decoders, input_embeds, kv_caches, pos):
    """
    Runs through the text decoders sequentially while updating the kv caches in place.

    Args:
        text_decoders: List of transformer decoder layers
        input_embeds: Initial token embeddings input
        kv_caches: Key-value cache states to be updated
        pos: Current sequence position for indexing into kv caches

    Returns:
        Tuple of (logits, hidden_states) for the final decoder layer
    """
    hidden, logits = input_embeds, None
    for i, decoder in enumerate(text_decoders):
        logits, hidden, kv_cache_update = decoder.run(
            None,
            {
                "inputs_embeds": hidden,
                "kv_cache": kv_caches[i][:, :, :, :, :pos, :],
            },
        )
        kv_caches[i][:, :, :, :, pos : pos + hidden.shape[-2], :] = kv_cache_update
    assert logits is not None  # at least one decoder must be present
    return logits, hidden


class VL:
    def __init__(
        self,
        model_path: Optional[str],
    ):
        """
        Initialize the Moondream VL (Vision Language) model.

        Args:
            model_path (str): The path to the model file.

        Returns:
            None
        """
        ort.set_default_logger_severity(3)

        if ort.get_device() == "GPU":
            ort_settings = {
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            }
        else:
            # Fall back to CPU if no GPU is available.
            ort_memory_info = ort.OrtMemoryInfo(
                "Cpu",
                ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR,
                0,
                ort.OrtMemType.DEFAULT,
            )
            ort.create_and_register_allocator(ort_memory_info, None)
            sess_options = ort.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            sess_options.add_session_config_entry("session.use_env_allocators", "1")
            ort_settings = {"sess_options": sess_options}

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
                elif name == "size_encoder.onnx":
                    self.size_encoder = ort.InferenceSession(contents, **ort_settings)
                elif name == "size_decoder.onnx":
                    self.size_decoder = ort.InferenceSession(contents, **ort_settings)
                elif name == "coord_encoder.onnx":
                    self.coord_encoder = ort.InferenceSession(contents, **ort_settings)
                elif name == "coord_decoder.onnx":
                    self.coord_decoder = ort.InferenceSession(contents, **ort_settings)
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

        # Run vision encoder.
        patch_emb = self.vision_encoder.run(None, {"input": image_patches})[0]
        patch_emb = np.concatenate([patch_emb[0], patch_emb[1]], axis=-1)
        patch_emb = np.expand_dims(patch_emb, axis=0)
        (inputs_embeds,) = self.vision_projection.run(None, {"input": patch_emb})

        # Run image embeddings through text decoders.
        kv_caches = self.initial_kv_caches
        pos = inputs_embeds.shape[-2] + kv_caches[0].shape[-2]
        for i, decoder in enumerate(self.text_decoders):
            _, inputs_embeds, kv_cache_update = decoder.run(
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
        kv_caches = prepare_kv_caches(self.initial_kv_caches, encoded_image)

        pos = encoded_image.pos
        generated_tokens = 0
        while generated_tokens < max_tokens:
            logits, hidden = run_decoders(self.text_decoders, hidden, kv_caches, pos)
            pos += hidden.shape[-2]

            next_token = np.argmax(logits, axis=-1)[0]
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

        prompt_toks = (
            self.templates["query"]["prefix"]
            + self.tokenizer.encode(question).ids
            + self.templates["query"]["suffix"]
        )

        (input_embeds,) = self.text_encoder.run(None, {"input_ids": [prompt_toks]})
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
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
    ) -> DetectOutput:
        """
        Detect and localize the specified object in the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be analyzed.
            object (str): The object to be detected in the image.

        Returns:
            List[Region]: A list of Region objects representing the detected instances of the specified object.
        """

        # Verify that the coord and size encoders and decoders are available.
        if not (
            hasattr(self, "coord_decoder")
            and hasattr(self, "coord_encoder")
            and hasattr(self, "size_decoder")
            and hasattr(self, "size_encoder")
        ):
            raise NotImplementedError("Model does not support 'detect'.")

        prompt_toks = (
            self.templates["detect"]["prefix"]
            + self.tokenizer.encode(object).ids
            + self.templates["detect"]["suffix"]
        )

        (hidden,) = self.text_encoder.run(None, {"input_ids": [prompt_toks]})
        encoded_image = self.encode_image(image)
        kv_caches = prepare_kv_caches(self.initial_kv_caches, encoded_image)

        objects = []
        pos = encoded_image.pos
        max_objects = 50

        # Greedy decoding: check if EOS is the most likely token, and break if so.
        # Otherwise, decode three tokens for the center coordinates and size.
        while len(objects) < max_objects:
            logits, hidden = run_decoders(self.text_decoders, hidden, kv_caches, pos)
            pos += hidden.shape[-2]

            if np.argmax(logits, axis=-1)[0] == self.special_tokens["eos"]:
                break

            # Decode and encode x center coordinate.
            (x_center,) = self.coord_decoder.run(None, {"input": hidden[0, -1, :]})
            x_center = np.argmax(x_center, axis=-1) / x_center.shape[-1]
            (hidden,) = self.coord_encoder.run(None, {"input": [x_center]})
            hidden = np.expand_dims(np.expand_dims(hidden, 0), 0)

            # Decode and encode y center coordinate.
            logits, hidden = run_decoders(self.text_decoders, hidden, kv_caches, pos)
            pos += hidden.shape[-2]
            (y_center,) = self.coord_decoder.run(None, {"input": hidden[0, -1, :]})
            y_center = np.argmax(y_center, axis=-1) / y_center.shape[-1]
            (hidden,) = self.coord_encoder.run(None, {"input": [y_center]})
            hidden = np.expand_dims(np.expand_dims(hidden, 0), 0)

            # Decode and encode size.
            logits, hidden = run_decoders(self.text_decoders, hidden, kv_caches, pos)
            pos += hidden.shape[-2]
            (size,) = self.size_decoder.run(None, {"input": hidden[0, -1, :]})
            w = np.argmax(size[0], axis=-1) / size.shape[-1]
            h = np.argmax(size[1], axis=-1) / size.shape[-1]
            (hidden,) = self.size_encoder.run(None, {"input": [w, h]})
            hidden = np.expand_dims(np.expand_dims(hidden, 0), 0)

            objects.append(
                {
                    "x_min": float(x_center - w / 2),
                    "y_min": float(y_center - h / 2),
                    "x_max": float(x_center + w / 2),
                    "y_max": float(y_center + h / 2),
                }
            )

        return {"objects": objects}
