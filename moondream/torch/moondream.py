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


SamplingSettings = TypedDict(
    "SamplingSettings",
    {"max_tokens": int},
    total=False,
)

DEFAULT_MAX_TOKENS = 512


def _min_p_sampler(
    logits: torch.Tensor,
    min_p: float = 0.1,
    filter_value: float = 0,
    min_tokens_to_keep: int = 1,
    temp=0.5,
) -> torch.Tensor:
    """
    Min-p sampler adapted from https://github.com/oobabooga/text-generation-webui/blob/3146124ec01f02c8fb1650a6517cf1b60b537aaf/modules/sampler_hijack.py#L16C17-L16C17
    https://arxiv.org/pdf/2407.01082
    """
    logits = logits / temp
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    scaled_min_p = min_p * top_probs
    tokens_to_remove = probs < scaled_min_p
    sorted_indices = torch.argsort(logits, descending=True, dim=-1)
    sorted_indices_to_remove = torch.gather(
        tokens_to_remove, dim=-1, index=sorted_indices
    )
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, filter_value)
    token = torch.multinomial(logits, num_samples=1)
    return token.squeeze(0)


@dataclass(frozen=True)
class EncodedImage:
    pos: int
    caches: List[Tuple[torch.Tensor, torch.Tensor]]


class KVCache(nn.Module):
    def __init__(self, n_heads, max_context, dim, device, dtype):
        super().__init__()
        cache_shape = (1, n_heads, max_context, dim // n_heads)
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
            "vikhyatk/moondream2", revision="2024-08-26"
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
        hidden = text_decoder(x[None], self.text, attn_mask, pos_ids, self.config.text)
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

    def _prefill_prompt(self, prompt_tokens: torch.Tensor, pos: int):
        with torch.inference_mode():
            prompt_emb = text_encoder(prompt_tokens, self.text)
            torch._dynamo.mark_dynamic(prompt_emb, 1)
            mask = self.attn_mask[:, :, pos : pos + prompt_emb.size(1), :]
            pos_ids = torch.arange(pos, pos + prompt_emb.size(1), dtype=torch.long)
            hidden = self._prefill(prompt_emb, mask, pos_ids)
            logits = lm_head(hidden, self.text)
            next_token = torch.argmax(logits, dim=-1)
        pos = pos + prompt_emb.size(1)
        return logits, hidden, next_token, pos

    def _generate_text(
        self,
        prompt_tokens: torch.Tensor,
        pos: int,
        max_tokens: int,
    ):
        _, _, next_token, pos = self._prefill_prompt(prompt_tokens, pos)

        def generator(next_token, pos):
            mask = torch.zeros(1, 1, 2048, device=self.device, dtype=torch.bool)
            mask[:, :, :pos] = 1
            pos_ids = torch.tensor([pos], device=self.device, dtype=torch.long)
            generated_tokens = 0

            while (
                next_token_id := next_token.item()
            ) != self.config.tokenizer.eos_id and generated_tokens < max_tokens:
                yield self.tokenizer.decode([next_token_id])

                with torch.inference_mode():
                    next_emb = text_encoder(next_token, self.text)
                    mask[:, :, pos], pos_ids[0] = 1, pos
                    logits, _ = self._decode_one_tok(next_emb, mask, pos_ids)
                    pos += 1
                    next_token = torch.argmax(logits, dim=-1)
                    generated_tokens += 1

        return generator(next_token, pos)

    def query(
        self,
        image: Union[Image.Image, EncodedImage],
        question: str,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ):
        if self.config.tokenizer.templates["query"] is None:
            raise NotImplementedError("Model does not support querying.")

        image = self.encode_image(image)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["query"]["prefix"]
                + self.tokenizer.encode(question).ids
                + self.config.tokenizer.templates["query"]["suffix"]
            ],
            device=self.device,
        )

        max_tokens = DEFAULT_MAX_TOKENS
        if settings:
            max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS)

        def generator():
            for token in self._generate_text(prompt_tokens, image.pos, max_tokens):
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
        length: Literal["normal", "short"] = "normal",
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
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

        max_tokens = DEFAULT_MAX_TOKENS
        if settings:
            max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS)

        def generator():
            for token in self._generate_text(prompt_tokens, image.pos, max_tokens):
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
        max_points: int = 50,
    ):
        out = []
        mask = torch.zeros(1, 1, 2048, device=self.device, dtype=torch.bool)
        mask[:, :, :pos] = 1
        pos_ids = torch.tensor([pos], device=self.device, dtype=torch.long)

        with torch.inference_mode():
            while (
                next_token.item() != self.config.tokenizer.eos_id
                and len(out) < max_points
            ):
                x_logits = decode_coordinate(hidden, self.region)
                x_center = torch.argmax(x_logits, dim=-1) / x_logits.size(-1)
                next_emb = encode_coordinate(
                    x_center.to(dtype=x_logits.dtype), self.region
                )

                # Decode y-coordinate
                mask[:, :, pos], pos_ids[0] = 1, pos
                _, hidden = self._decode_one_tok(next_emb, mask, pos_ids)
                pos += 1
                y_logits = decode_coordinate(hidden, self.region)
                y_center = torch.argmax(y_logits, dim=-1) / y_logits.size(-1)
                next_emb = encode_coordinate(
                    y_center.to(dtype=y_logits.dtype), self.region
                )

                # Decode size
                if include_size:
                    mask[:, :, pos], pos_ids[0] = 1, pos
                    logits, hidden = self._decode_one_tok(next_emb, mask, pos_ids)
                    pos += 1
                    size_logits = decode_size(hidden, self.region)
                    w = torch.argmax(size_logits[0], dim=-1) / size_logits.size(-1)
                    h = torch.argmax(size_logits[1], dim=-1) / size_logits.size(-1)
                    next_emb = encode_size(
                        torch.tensor(
                            [w, h], device=self.device, dtype=size_logits.dtype
                        ),
                        self.region,
                    )[None]

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
        settings: Optional[SamplingSettings] = None,
    ):
        if self.config.tokenizer.templates["detect"] is None:
            raise NotImplementedError("Model does not support object detection.")

        image = self.encode_image(image)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["detect"]["prefix"]
                + self.tokenizer.encode(object).ids
                + self.config.tokenizer.templates["detect"]["suffix"]
            ],
            device=self.device,
        )

        _, hidden, next_token, pos = self._prefill_prompt(prompt_tokens, image.pos)
        hidden = hidden[:, -1:, :]

        objects = self._generate_points(
            hidden, next_token, pos, include_size=True, max_points=50
        )

        return {"objects": objects}

    def point(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[SamplingSettings] = None,
    ):
        if self.config.tokenizer.templates["point"] is None:
            raise NotImplementedError("Model does not support pointing.")

        image = self.encode_image(image)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["point"]["prefix"]
                + self.tokenizer.encode(object).ids
                + self.config.tokenizer.templates["point"]["suffix"]
            ],
            device=self.device,
        )

        _, hidden, next_token, pos = self._prefill_prompt(prompt_tokens, image.pos)
        hidden = hidden[:, -1:, :]

        objects = self._generate_points(
            hidden, next_token, pos, include_size=False, max_points=50
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
                hidden, next_token, pos, include_size=False, max_points=1
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
