import torch
import torch.nn as nn
import random

from typing import Literal, Tuple, TypedDict, Union, Dict, Any, Optional, List
from PIL import Image
from dataclasses import dataclass
from tokenizers import Tokenizer

from .config import MoondreamConfig
from .image_crops import reconstruct_from_crops
from .vision import vision_encoder, vision_projection, prepare_crops, build_vision_model
from .text import build_text_model, text_encoder, lm_head, text_decoder
from .region import decode_coordinate, encode_coordinate, decode_size, encode_size
from .utils import remove_outlier_points


TextSamplingSettings = TypedDict(
    "TextSamplingSettings",
    {
        "max_tokens": int,
        "temperature": float,
        "top_p": float,
    },
    total=False,
)

ObjectSamplingSettings = TypedDict(
    "ObjectSamplingSettings",
    {"max_objects": int},
    total=False,
)

DEFAULT_MAX_TOKENS = 768
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.3
DEFAULT_MAX_OBJECTS = 50


@dataclass(frozen=True)
class EncodedImage:
    pos: int
    caches: List[Tuple[torch.Tensor, torch.Tensor]]


class KVCache(nn.Module):

    def __init__(self, n_heads, n_kv_heads, max_context, dim, device, dtype):
        super().__init__()
        cache_shape = (1, n_kv_heads, max_context, dim // n_heads)
        self.register_buffer(
            "k_cache", torch.zeros(*cache_shape, device=device, dtype=dtype)
        )
        self.register_buffer(
            "v_cache", torch.zeros(*cache_shape, device=device, dtype=dtype)
        )

    def update(self, pos_ids, k, v):
        kout, vout = self.k_cache, self.v_cache
        kout[:, :, pos_ids, :] = k
        vout[:, :, pos_ids, :] = v
        return kout, vout


class MoondreamModel(nn.Module):
    def __init__(self, config: MoondreamConfig, dtype=torch.float16, setup_caches=True):
        super().__init__()
        self.config = config

        self.tokenizer = Tokenizer.from_pretrained(
            "vikhyatk/moondream2", revision="2025-01-09"
        )
        self.vision = build_vision_model(config.vision, dtype)
        self.text = build_text_model(config.text, dtype)

        # Region Model
        self.region = nn.ModuleDict(
            {
                "coord_encoder": nn.Linear(
                    config.region.coord_feat_dim, config.region.dim, dtype=dtype
                ),
                "coord_decoder": nn.ModuleDict(
                    {
                        "fc1": nn.Linear(
                            config.region.dim, config.region.inner_dim, dtype=dtype
                        ),
                        "fc2": nn.Linear(
                            config.region.inner_dim,
                            config.region.coord_out_dim,
                            dtype=dtype,
                        ),
                    }
                ),
                "size_encoder": nn.Linear(
                    config.region.size_feat_dim, config.region.dim, dtype=dtype
                ),
                "size_decoder": nn.ModuleDict(
                    {
                        "fc1": nn.Linear(
                            config.region.dim, config.region.inner_dim, dtype=dtype
                        ),
                        "fc2": nn.Linear(
                            config.region.inner_dim,
                            config.region.size_out_dim,
                            dtype=dtype,
                        ),
                    }
                ),
            }
        )
        self.region.coord_features = nn.Parameter(
            torch.empty(config.region.coord_feat_dim // 2, 1, dtype=dtype).T
        )
        self.region.size_features = nn.Parameter(
            torch.empty(config.region.size_feat_dim // 2, 2, dtype=dtype).T
        )

        attn_mask = torch.tril(
            torch.ones(
                1, 1, config.text.max_context, config.text.max_context, dtype=torch.bool
            )
        )
        patch_w = config.vision.crop_size // config.vision.enc_patch_size
        prefix_attn_len = 1 + patch_w**2
        attn_mask[..., :prefix_attn_len, :prefix_attn_len] = 1
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        # Initialize KV caches.
        if setup_caches:
            self._setup_caches()

    def _setup_caches(self):
        c = self.config.text
        for b in self.text.blocks:
            b.kv_cache = KVCache(
                c.n_heads,
                c.n_kv_heads,
                c.max_context,
                c.dim,
                device=self.device,
                dtype=self.vision.pos_emb.dtype,
            )

    @property
    def device(self):
        return self.vision.pos_emb.device

    def _vis_enc(self, x: torch.Tensor):
        return vision_encoder(x, self.vision, self.config.vision)

    def _vis_proj(self, g: torch.Tensor, r: torch.Tensor):
        return vision_projection(g, r, self.vision, self.config.vision)

    def _prefill(self, x: torch.Tensor, attn_mask: torch.Tensor, pos_ids: torch.Tensor):
        return text_decoder(x, self.text, attn_mask, pos_ids, self.config.text)

    def _decode_one_tok(
        self, x: torch.Tensor, attn_mask: torch.Tensor, pos_ids: torch.Tensor
    ):
        hidden = text_decoder(x, self.text, attn_mask, pos_ids, self.config.text)
        logits = lm_head(hidden, self.text)
        return logits, hidden

    def compile(self):
        # TODO: vision_projection is not being compiled
        self._vis_enc = torch.compile(self._vis_enc, fullgraph=True)
        self._prefill = torch.compile(self._prefill, fullgraph=True)
        self._decode_one_tok = torch.compile(
            self._decode_one_tok, fullgraph=True, mode="reduce-overhead"
        )

    def _run_vision_encoder(self, image: Image.Image) -> torch.Tensor:
        all_crops, tiling = prepare_crops(image, self.config.vision, device=self.device)
        torch._dynamo.mark_dynamic(all_crops, 0)

        outputs = self._vis_enc(all_crops)

        global_features = outputs[0]
        local_features = outputs[1:].view(
            -1,
            self.config.vision.enc_n_layers,
            self.config.vision.enc_n_layers,
            self.config.vision.enc_dim,
        )

        reconstructed = reconstruct_from_crops(
            local_features,
            tiling,
            patch_size=1,
            overlap_margin=self.config.vision.overlap_margin,
        )

        return self._vis_proj(global_features, reconstructed)

    def encode_image(self, image: Union[Image.Image, EncodedImage]) -> EncodedImage:
        if isinstance(image, EncodedImage):
            return image
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image or EncodedImage")

        # Run through text model in addition to the vision encoder, to minimize
        # re-computation if multiple queries are performed on this image.
        with torch.inference_mode():
            img_emb = self._run_vision_encoder(image)
            bos_emb = text_encoder(
                torch.tensor([[self.config.tokenizer.bos_id]], device=self.device),
                self.text,
            )
            inputs_embeds = torch.cat([bos_emb, img_emb[None]], dim=1)
            mask = self.attn_mask[:, :, 0 : inputs_embeds.size(1), :]
            pos_ids = torch.arange(inputs_embeds.size(1), dtype=torch.long)
            self._prefill(inputs_embeds, mask, pos_ids)

        return EncodedImage(
            pos=inputs_embeds.size(1),
            caches=[
                (
                    b.kv_cache.k_cache[:, :, : inputs_embeds.size(1), :].clone(),
                    b.kv_cache.v_cache[:, :, : inputs_embeds.size(1), :].clone(),
                )
                for b in self.text.blocks
            ],
        )

    def _apply_top_p(self, probs: torch.Tensor, top_p: float):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_probs = torch.zeros_like(probs)
        next_probs.scatter_(dim=-1, index=probs_idx, src=probs_sort)
        return next_probs

    def _prefill_prompt(
        self, prompt_tokens: torch.Tensor, pos: int, temperature: float, top_p: float
    ):
        with torch.inference_mode():
            prompt_emb = text_encoder(prompt_tokens, self.text)
            torch._dynamo.mark_dynamic(prompt_emb, 1)
            mask = self.attn_mask[:, :, pos : pos + prompt_emb.size(1), :]
            pos_ids = torch.arange(pos, pos + prompt_emb.size(1), dtype=torch.long)
            hidden = self._prefill(prompt_emb, mask, pos_ids)
            logits = lm_head(hidden, self.text)

            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                probs = self._apply_top_p(probs, top_p)
                next_token = torch.multinomial(probs, num_samples=1)

        pos = pos + prompt_emb.size(1)
        return logits, hidden, next_token, pos

    def _generate_text(
        self,
        prompt_tokens: torch.Tensor,
        pos: int,
        settings: Optional[TextSamplingSettings] = None,
    ):
        max_tokens = (
            settings.get("max_tokens", DEFAULT_MAX_TOKENS)
            if settings
            else DEFAULT_MAX_TOKENS
        )
        temperature = (
            settings.get("temperature", DEFAULT_TEMPERATURE)
            if settings
            else DEFAULT_TEMPERATURE
        )
        top_p = settings.get("top_p", DEFAULT_TOP_P) if settings else DEFAULT_TOP_P

        _, _, next_token, pos = self._prefill_prompt(
            prompt_tokens, pos, temperature, top_p
        )

        def generator(next_token, pos):
            mask = torch.zeros(1, 1, 2048, device=self.device, dtype=torch.bool)
            mask[:, :, :pos] = 1
            pos_ids = torch.tensor([pos], device=self.device, dtype=torch.long)
            generated_tokens = 0

            # For properly handling token streaming with Unicode
            token_cache = []
            print_len = 0

            while (
                next_token_id := next_token.item()
            ) != self.config.tokenizer.eos_id and generated_tokens < max_tokens:
                # Add token to our cache
                token_cache.append(next_token_id)

                # Decode all tokens collected so far
                text = self.tokenizer.decode(token_cache)

                # After a newline, we flush the cache completely
                if text.endswith("\n"):
                    printable_text = text[print_len:]
                    token_cache = []
                    print_len = 0
                    if printable_text:
                        yield printable_text
                # If the last token is a CJK character, we can safely print it
                elif len(text) > 0 and _is_cjk_char(ord(text[-1])):
                    printable_text = text[print_len:]
                    print_len += len(printable_text)
                    if printable_text:
                        yield printable_text
                # Otherwise, only print up to the last space to avoid cutting words
                else:
                    last_space_idx = text.rfind(" ", print_len)
                    if last_space_idx >= print_len:
                        printable_text = text[print_len : last_space_idx + 1]
                        print_len += len(printable_text)
                        if printable_text:
                            yield printable_text

                with torch.inference_mode():
                    next_emb = text_encoder(next_token, self.text)
                    mask[:, :, pos], pos_ids[0] = 1, pos
                    logits, _ = self._decode_one_tok(next_emb, mask, pos_ids)
                    pos += 1

                    if temperature == 0:
                        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)  # (1, 1)
                    else:
                        probs = torch.softmax(logits / temperature, dim=-1)  # (1, V)
                        probs = self._apply_top_p(probs, top_p)
                        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                    generated_tokens += 1

            # Flush any remaining text in the cache
            if token_cache:
                text = self.tokenizer.decode(token_cache)
                printable_text = text[print_len:]
                if printable_text:
                    yield printable_text

        return generator(next_token, pos)

    def query(
        self,
        image: Union[Image.Image, EncodedImage],
        question: str,
        stream: bool = False,
        settings: Optional[TextSamplingSettings] = None,
    ):
        if self.config.tokenizer.templates["query"] is None:
            raise NotImplementedError("Model does not support querying.")

        image = self.encode_image(image)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["query"]["prefix"]
                + self.tokenizer.encode(" " + question).ids
                + self.config.tokenizer.templates["query"]["suffix"]
            ],
            device=self.device,
        )

        def generator():
            for token in self._generate_text(prompt_tokens, image.pos, settings):
                yield token

        if stream:
            return {"answer": generator()}
        else:
            return {"answer": "".join(list(generator()))}

    def load_encoded_image(self, encoded_image: EncodedImage):
        for b, (k, v) in zip(self.text.blocks, encoded_image.caches):
            b.kv_cache.k_cache[:, :, : k.size(2), :] = k
            b.kv_cache.v_cache[:, :, : v.size(2), :] = v

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: Literal["normal", "short", "long"] = "normal",
        stream: bool = False,
        settings: Optional[TextSamplingSettings] = None,
    ):
        if self.config.tokenizer.templates["caption"] is None:
            raise NotImplementedError("Model does not support captioning.")
        if length not in self.config.tokenizer.templates["caption"]:
            raise ValueError(f"Model does not support caption length '{length}'.")

        image = self.encode_image(image)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [self.config.tokenizer.templates["caption"][length]], device=self.device
        )

        def generator():
            for token in self._generate_text(prompt_tokens, image.pos, settings):
                yield token

        if stream:
            return {"caption": generator()}
        else:
            return {"caption": "".join(list(generator()))}

    def _generate_points(
        self,
        hidden: torch.Tensor,
        next_token: torch.Tensor,
        pos: int,
        include_size: bool = True,
        max_objects: int = DEFAULT_MAX_OBJECTS,
    ):
        out = []
        mask = torch.zeros(1, 1, 2048, device=self.device, dtype=torch.bool)
        mask[:, :, :pos] = 1
        pos_ids = torch.tensor([pos], device=self.device, dtype=torch.long)

        with torch.inference_mode():
            while (
                next_token.item() != self.config.tokenizer.eos_id
                and len(out) < max_objects
            ):
                x_logits = decode_coordinate(hidden, self.region)
                x_center = torch.argmax(x_logits, dim=-1) / x_logits.size(-1)
                next_emb = encode_coordinate(
                    x_center.to(dtype=x_logits.dtype), self.region
                ).unsqueeze(0)

                # Decode y-coordinate
                mask[:, :, pos], pos_ids[0] = 1, pos
                _, hidden = self._decode_one_tok(next_emb, mask, pos_ids)
                pos += 1
                y_logits = decode_coordinate(hidden, self.region)
                y_center = torch.argmax(y_logits, dim=-1) / y_logits.size(-1)
                next_emb = encode_coordinate(
                    y_center.to(dtype=y_logits.dtype), self.region
                ).unsqueeze(0)

                # Decode size
                if include_size:
                    mask[:, :, pos], pos_ids[0] = 1, pos
                    logits, hidden = self._decode_one_tok(next_emb, mask, pos_ids)
                    pos += 1
                    size_logits = decode_size(hidden, self.region)

                    # Get bin indices from the logits
                    w_bin = torch.argmax(size_logits[0], dim=-1)
                    h_bin = torch.argmax(size_logits[1], dim=-1)

                    # Convert from bin indices to actual size values using the inverse of the log-scale mapping
                    # Formula: size = 2^((bin / 1023.0) * 10.0 - 10.0)
                    w = torch.pow(2.0, (w_bin.float() / 1023.0) * 10.0 - 10.0)
                    h = torch.pow(2.0, (h_bin.float() / 1023.0) * 10.0 - 10.0)

                    next_emb = (
                        encode_size(
                            torch.tensor(
                                [w, h], device=self.device, dtype=size_logits.dtype
                            ),
                            self.region,
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )

                    # Add object
                    out.append(
                        {
                            "x_min": x_center.item() - w.item() / 2,
                            "y_min": y_center.item() - h.item() / 2,
                            "x_max": x_center.item() + w.item() / 2,
                            "y_max": y_center.item() + h.item() / 2,
                        }
                    )
                else:
                    out.append({"x": x_center.item(), "y": y_center.item()})

                # Decode next token (x-coordinate, or eos)
                mask[:, :, pos], pos_ids[0] = 1, pos
                logits, hidden = self._decode_one_tok(next_emb, mask, pos_ids)
                pos += 1
                next_token = torch.argmax(logits, dim=-1)

        return out

    def detect(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[ObjectSamplingSettings] = None,
    ):
        if self.config.tokenizer.templates["detect"] is None:
            raise NotImplementedError("Model does not support object detection.")

        image = self.encode_image(image)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["detect"]["prefix"]
                + self.tokenizer.encode(" " + object).ids
                + self.config.tokenizer.templates["detect"]["suffix"]
            ],
            device=self.device,
        )

        _, hidden, next_token, pos = self._prefill_prompt(
            prompt_tokens, image.pos, temperature=0, top_p=0
        )
        hidden = hidden[:, -1:, :]

        max_objects = (
            settings.get("max_objects", DEFAULT_MAX_OBJECTS)
            if settings
            else DEFAULT_MAX_OBJECTS
        )
        objects = self._generate_points(
            hidden, next_token, pos, include_size=True, max_objects=max_objects
        )

        return {"objects": objects}

    def point(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[ObjectSamplingSettings] = None,
    ):
        if self.config.tokenizer.templates["point"] is None:
            raise NotImplementedError("Model does not support pointing.")

        image = self.encode_image(image)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["point"]["prefix"]
                + self.tokenizer.encode(" " + object).ids
                + self.config.tokenizer.templates["point"]["suffix"]
            ],
            device=self.device,
        )

        _, hidden, next_token, pos = self._prefill_prompt(
            prompt_tokens, image.pos, temperature=0, top_p=0
        )
        hidden = hidden[:, -1:, :]

        max_objects = (
            settings.get("max_objects", DEFAULT_MAX_OBJECTS)
            if settings
            else DEFAULT_MAX_OBJECTS
        )
        objects = self._generate_points(
            hidden, next_token, pos, include_size=False, max_objects=max_objects
        )

        return {"points": objects}

    def _detect_gaze(
        self,
        image: EncodedImage,
        source: Tuple[float, float],
        force_detect: bool = False,
    ):
        with torch.inference_mode():
            before_emb = text_encoder(
                torch.tensor(
                    [self.tokenizer.encode("\n\nPoint:").ids], device=self.device
                ),
                self.text,
            )
            after_emb = text_encoder(
                torch.tensor(
                    [self.tokenizer.encode(" gaze\n\n").ids], device=self.device
                ),
                self.text,
            )
            x_emb = encode_coordinate(
                torch.tensor([[[source[0]]]], device=self.device, dtype=torch.float16),
                self.region,
            )
            y_emb = encode_coordinate(
                torch.tensor([[[source[1]]]], device=self.device, dtype=torch.float16),
                self.region,
            )

            prompt_emb = torch.cat([before_emb, x_emb, y_emb, after_emb], dim=1)

            self.load_encoded_image(image)

            mask = self.attn_mask[:, :, image.pos : image.pos + prompt_emb.size(1), :]
            pos_ids = torch.arange(
                image.pos, image.pos + prompt_emb.size(1), dtype=torch.long
            )
            hidden = self._prefill(prompt_emb, mask, pos_ids)
            logits = lm_head(hidden, self.text)
            next_token = torch.argmax(logits, dim=-1)
            pos = image.pos + prompt_emb.size(1)
            hidden = hidden[:, -1:, :]

            if force_detect:
                next_token = torch.tensor([[0]], device=self.device)

            if next_token.item() == self.config.tokenizer.eos_id:
                return None

            gaze = self._generate_points(
                hidden, next_token, pos, include_size=False, max_objects=1
            )
            return gaze[0]

    def detect_gaze(
        self,
        image: Union[Image.Image, EncodedImage],
        eye: Optional[Tuple[float, float]] = None,
        face: Optional[Dict[str, float]] = None,
        unstable_settings: Dict[str, Any] = {},
    ):
        if "force_detect" in unstable_settings:
            force_detect = unstable_settings["force_detect"]
        else:
            force_detect = False

        if "prioritize_accuracy" in unstable_settings:
            prioritize_accuracy = unstable_settings["prioritize_accuracy"]
        else:
            prioritize_accuracy = False

        if not prioritize_accuracy:
            if eye is None:
                raise ValueError("eye must be provided when prioritize_accuracy=False")
            image = self.encode_image(image)
            return {"gaze": self._detect_gaze(image, eye, force_detect=force_detect)}
        else:
            if (
                not isinstance(image, Image.Image)
                and "flip_enc_img" not in unstable_settings
            ):
                raise ValueError(
                    "image must be a PIL Image when prioritize_accuracy=True, "
                    "or flip_enc_img must be provided"
                )
            if face is None:
                raise ValueError("face must be provided when prioritize_accuracy=True")

            encoded_image = self.encode_image(image)
            if (
                isinstance(image, Image.Image)
                and "flip_enc_img" not in unstable_settings
            ):
                flipped_pil = image.copy()
                flipped_pil = flipped_pil.transpose(method=Image.FLIP_LEFT_RIGHT)
                encoded_flipped_image = self.encode_image(flipped_pil)
            else:
                encoded_flipped_image = unstable_settings["flip_enc_img"]

            N = 10

            detections = [
                self._detect_gaze(
                    encoded_image,
                    (
                        random.uniform(face["x_min"], face["x_max"]),
                        random.uniform(face["y_min"], face["y_max"]),
                    ),
                    force_detect=force_detect,
                )
                for _ in range(N)
            ]
            detections = [
                (gaze["x"], gaze["y"]) for gaze in detections if gaze is not None
            ]
            flipped_detections = [
                self._detect_gaze(
                    encoded_flipped_image,
                    (
                        1 - random.uniform(face["x_min"], face["x_max"]),
                        random.uniform(face["y_min"], face["y_max"]),
                    ),
                    force_detect=force_detect,
                )
                for _ in range(N)
            ]
            detections.extend(
                [
                    (1 - gaze["x"], gaze["y"])
                    for gaze in flipped_detections
                    if gaze is not None
                ]
            )

            if len(detections) < N:
                return {"gaze": None}

            detections = remove_outlier_points(detections)
            mean_gaze = (
                sum(gaze[0] for gaze in detections) / len(detections),
                sum(gaze[1] for gaze in detections) / len(detections),
            )

            return {"gaze": {"x": mean_gaze[0], "y": mean_gaze[1]}}


def _is_cjk_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
        return True
    return False
