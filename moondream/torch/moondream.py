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
from .region import (
    decode_coordinate,
    encode_coordinate,
    decode_size,
    encode_size,
    encode_spatial_refs,
    SpatialRefs,
)
from .layers import QuantizedLinear
from .lora import variant_state_dict
from .utils import remove_outlier_points

ImageEncodingSettings = TypedDict(
    "ImageEncodingSettings",
    {"variant": str},
    total=False,
)

TextSamplingSettings = TypedDict(
    "TextSamplingSettings",
    {
        "max_tokens": int,
        "temperature": float,
        "top_p": float,
        "variant": str,
    },
    total=False,
)

ObjectSamplingSettings = TypedDict(
    "ObjectSamplingSettings",
    {"max_objects": int, "variant": str},
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

    def __init__(
        self, config: MoondreamConfig, dtype=torch.bfloat16, setup_caches=True
    ):
        super().__init__()
        self.config = config

        self.tokenizer = Tokenizer.from_pretrained("moondream/starmie-v1")
        self.vision = build_vision_model(config.vision, dtype)
        self.text = build_text_model(config.text, dtype)

        # Region Model
        linear_cls = (
            QuantizedLinear if config.region.group_size is not None else nn.Linear
        )
        self.region = nn.ModuleDict(
            {
                "coord_encoder": linear_cls(
                    config.region.coord_feat_dim, config.region.dim, dtype=dtype
                ),
                "coord_decoder": nn.ModuleDict(
                    {
                        "fc1": linear_cls(
                            config.region.dim, config.region.inner_dim, dtype=dtype
                        ),
                        "fc2": linear_cls(
                            config.region.inner_dim,
                            config.region.coord_out_dim,
                            dtype=dtype,
                        ),
                    }
                ),
                "size_encoder": linear_cls(
                    config.region.size_feat_dim, config.region.dim, dtype=dtype
                ),
                "size_decoder": nn.ModuleDict(
                    {
                        "fc1": linear_cls(
                            config.region.dim, config.region.inner_dim, dtype=dtype
                        ),
                        "fc2": linear_cls(
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

    def _prefill(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        lora: Optional[torch.Tensor],
    ):
        return text_decoder(x, self.text, attn_mask, pos_ids, self.config.text, lora)

    def _decode_one_tok(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        lora: Optional[torch.Tensor],
    ):
        hidden = text_decoder(x, self.text, attn_mask, pos_ids, self.config.text, lora)
        logits = lm_head(hidden, self.text)
        return logits, hidden

    def compile(self):
        for module in self.modules():
            if isinstance(module, QuantizedLinear):
                module.unpack()

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

    def encode_image(
        self,
        image: Union[Image.Image, EncodedImage],
        settings: Optional[ImageEncodingSettings] = None,
    ) -> EncodedImage:
        if isinstance(image, EncodedImage):
            return image
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image or EncodedImage")

        lora = (
            variant_state_dict(settings["variant"], device=self.device)
            if settings is not None and settings["variant"] is not None
            else None
        )

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
            self._prefill(inputs_embeds, mask, pos_ids, lora)

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
        self,
        prompt_tokens: torch.Tensor,
        pos: int,
        temperature: float,
        top_p: float,
        spatial_refs: Optional[SpatialRefs] = None,
        attn_mask: Optional[torch.Tensor] = None,
        lora: Optional[dict] = None,
    ):
        with torch.inference_mode():
            prompt_emb = text_encoder(prompt_tokens, self.text)

            if spatial_refs:
                encoded_refs = encode_spatial_refs(spatial_refs, self.region)
                prompt_emb[prompt_tokens == self.config.tokenizer.coord_id] = (
                    encoded_refs["coords"]
                )
                if encoded_refs["sizes"] is not None:
                    prompt_emb[prompt_tokens == self.config.tokenizer.size_id] = (
                        encoded_refs["sizes"]
                    )

            torch._dynamo.mark_dynamic(prompt_emb, 1)

            if attn_mask is None:
                attn_mask = self.attn_mask

            mask = attn_mask[:, :, pos : pos + prompt_emb.size(1), :]
            pos_ids = torch.arange(pos, pos + prompt_emb.size(1), dtype=torch.long)
            hidden_BC = self._prefill(prompt_emb, mask, pos_ids, lora)
            logits_BV = lm_head(hidden_BC, self.text)

            if temperature == 0:
                next_token = torch.argmax(logits_BV, dim=-1).unsqueeze(1)
            else:
                probs = torch.softmax(logits_BV / temperature, dim=-1)
                probs = self._apply_top_p(probs, top_p)
                next_token = torch.multinomial(probs, num_samples=1)

        pos = pos + prompt_emb.size(1)
        return logits_BV, hidden_BC, next_token, pos

    def _generate_reasoning(
        self,
        prompt_tokens,
        pos,
        settings: Optional[TextSamplingSettings] = None,
        spatial_refs: Optional[SpatialRefs] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, str, List[dict]]:
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
        lora = (
            variant_state_dict(settings["variant"], device=self.device)
            if settings is not None and "variant" in settings
            else None
        )

        top_p = settings.get("top_p", DEFAULT_TOP_P) if settings else DEFAULT_TOP_P
        eos_id = self.config.tokenizer.answer_id

        _, last_hidden_BC, next_token, pos = self._prefill_prompt(
            prompt_tokens,
            pos,
            temperature,
            top_p,
            spatial_refs,
            attn_mask=attn_mask,
            lora=lora,
        )

        text_token_chunks = [[]]
        grounding_chunks = [[]]

        mask = torch.zeros(1, 1, 2048, device=self.device, dtype=torch.bool)
        mask[:, :, :pos] = 1
        pos_ids = torch.tensor([pos], device=self.device, dtype=torch.long)
        generated_tokens = 0

        while (
            next_token_id := next_token.item()
        ) != eos_id and generated_tokens < max_tokens:
            if (
                next_token_id == self.config.tokenizer.start_ground_points_id
                or next_token_id == self.config.tokenizer.end_ground_id
            ):
                text_token_chunks.append([])
                grounding_chunks.append([])

            text_token_chunks[-1].append(next_token_id)

            with torch.inference_mode():
                if next_token_id == self.config.tokenizer.coord_id:
                    coord_logits = decode_coordinate(last_hidden_BC, self.region)
                    coord = torch.argmax(coord_logits, dim=-1) / coord_logits.size(-1)
                    grounding_chunks[-1].append(coord.item())

                    next_emb = encode_coordinate(
                        coord.to(dtype=coord_logits.dtype), self.region
                    ).unsqueeze(0)
                else:
                    next_emb = text_encoder(next_token, self.text)

                mask[:, :, pos], pos_ids[0] = 1, pos

                logits_BV, last_hidden_BC = self._decode_one_tok(
                    next_emb, mask, pos_ids, lora
                )
                logits_BV[:, self.config.tokenizer.eos_id] = float("-inf")
                logits_BV[:, self.config.tokenizer.size_id] = float("-inf")

                pos += 1

                if temperature == 0:
                    next_token = torch.argmax(logits_BV, dim=-1).unsqueeze(1)  # (1, 1)
                else:
                    probs = torch.softmax(logits_BV / temperature, dim=-1)  # (1, V)
                    probs = self._apply_top_p(probs, top_p)
                    next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                generated_tokens += 1

        text_chunks = [
            self.tokenizer.decode(chunk_tokens) for chunk_tokens in text_token_chunks
        ]
        text = "".join(text_chunks)

        start_idx = 0
        grounding = []
        for text_chunk, grounding_chunk in zip(text_chunks, grounding_chunks):
            if len(grounding_chunk) > 1:
                points = []
                for i in range(0, len(grounding_chunk) - (len(grounding_chunk) % 2), 2):
                    points.append((grounding_chunk[i], grounding_chunk[i + 1]))
                grounding.append(
                    {
                        "start_idx": start_idx,
                        "end_idx": start_idx + len(text_chunk),
                        "points": points,
                    }
                )
            start_idx += len(text_chunk)

        return pos, text, grounding

    def _generate_answer(
        self,
        prompt_tokens: torch.Tensor,
        pos: int,
        settings: Optional[TextSamplingSettings] = None,
        spatial_refs: Optional[SpatialRefs] = None,
        eos_id: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
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
        eos_id = eos_id if eos_id is not None else self.config.tokenizer.eos_id
        lora = (
            variant_state_dict(settings["variant"], device=self.device)
            if settings is not None and "variant" in settings
            else None
        )

        _, _, next_token, pos = self._prefill_prompt(
            prompt_tokens,
            pos,
            temperature,
            top_p,
            spatial_refs,
            attn_mask=attn_mask,
            lora=lora,
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
            ) != eos_id and generated_tokens < max_tokens:
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
                # Otherwise, only yield up to the last space to avoid cutting words
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

                    logits_BV, _ = self._decode_one_tok(next_emb, mask, pos_ids, lora)
                    logits_BV[:, self.config.tokenizer.answer_id] = float("-inf")

                    pos += 1

                    if temperature == 0:
                        next_token = torch.argmax(logits_BV, dim=-1).unsqueeze(
                            1
                        )  # (1, 1)
                    else:
                        probs = torch.softmax(logits_BV / temperature, dim=-1)  # (1, V)
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
        image: Optional[Union[Image.Image, EncodedImage]] = None,
        question: str = None,
        reasoning: bool = False,
        spatial_refs: Optional[SpatialRefs] = None,
        stream: bool = False,
        settings: Optional[TextSamplingSettings] = None,
    ):
        if self.config.tokenizer.templates["query"] is None:
            raise NotImplementedError("Model does not support querying.")

        if question is None:
            raise ValueError("question must be provided.")

        if spatial_refs and image is None:
            raise ValueError("spatial_refs can only be used with an image.")

        attn_mask = self.attn_mask
        if image is not None:
            image = self.encode_image(image, settings)
            self.load_encoded_image(image)
            pos = image.pos
            prompt_toks = self.config.tokenizer.templates["query"]["prefix"]
        else:
            self._setup_caches()
            pos = 0
            prompt_toks = [
                self.config.tokenizer.bos_id
            ] + self.config.tokenizer.templates["query"]["prefix"]
            max_context = self.config.text.max_context
            attn_mask = torch.tril(
                torch.ones(1, 1, max_context, max_context, dtype=torch.bool)
            ).to(self.device)

        spatial_toks = []
        if spatial_refs:
            for ref in spatial_refs:
                coord_id = self.config.tokenizer.coord_id
                size_id = self.config.tokenizer.size_id
                if len(ref) == 2:
                    spatial_toks.extend([coord_id, coord_id])
                else:
                    spatial_toks.extend([coord_id, coord_id, size_id])

        prompt_tokens = [
            prompt_toks
            + spatial_toks
            + self.tokenizer.encode(question).ids
            + self.config.tokenizer.templates["query"]["suffix"]
        ]

        if reasoning:
            prompt_tokens[0] += [self.config.tokenizer.thinking_id]
            prompt_tokens = torch.tensor(prompt_tokens, device=self.device)
            pos, reasoning_text, reasoning_grounding = self._generate_reasoning(
                prompt_tokens, pos, settings, spatial_refs, attn_mask=attn_mask
            )
            prompt_tokens = [self.config.tokenizer.templates["query"]["suffix"]]
            reasoning_dict = {
                "reasoning": {"text": reasoning_text, "grounding": reasoning_grounding}
            }
        else:
            prompt_tokens[0] += self.config.tokenizer.templates["query"]["suffix"]
            reasoning_dict = {}

        prompt_tokens = torch.tensor(prompt_tokens, device=self.device)

        def generator():
            for token in self._generate_answer(
                prompt_tokens, pos, settings, spatial_refs, attn_mask=attn_mask
            ):
                yield token

        if stream:
            return {**reasoning_dict, "answer": generator()}
        else:
            return {**reasoning_dict, "answer": "".join(list(generator()))}

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

        image = self.encode_image(image, settings)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [self.config.tokenizer.templates["caption"][length]], device=self.device
        )

        def generator():
            for token in self._generate_answer(prompt_tokens, image.pos, settings):
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
        lora: Optional[dict] = None,
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
                _, hidden = self._decode_one_tok(next_emb, mask, pos_ids, lora)
                pos += 1
                y_logits = decode_coordinate(hidden, self.region)
                y_center = torch.argmax(y_logits, dim=-1) / y_logits.size(-1)
                next_emb = encode_coordinate(
                    y_center.to(dtype=y_logits.dtype), self.region
                ).unsqueeze(0)

                # Decode size
                if include_size:
                    mask[:, :, pos], pos_ids[0] = 1, pos
                    logits, hidden = self._decode_one_tok(next_emb, mask, pos_ids, lora)
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
                logits, hidden = self._decode_one_tok(next_emb, mask, pos_ids, lora)
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

        image = self.encode_image(image, settings)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["detect"]["prefix"]
                + self.tokenizer.encode(" " + object).ids
                + self.config.tokenizer.templates["detect"]["suffix"]
            ],
            device=self.device,
        )

        lora = (
            variant_state_dict(settings["variant"], device=self.device)
            if settings is not None and "variant" in settings
            else None
        )

        _, hidden, next_token, pos = self._prefill_prompt(
            prompt_tokens, image.pos, temperature=0, top_p=0, lora=lora
        )
        hidden = hidden[:, -1:, :]

        max_objects = (
            settings.get("max_objects", DEFAULT_MAX_OBJECTS)
            if settings
            else DEFAULT_MAX_OBJECTS
        )
        objects = self._generate_points(
            hidden,
            next_token,
            pos,
            include_size=True,
            max_objects=max_objects,
            lora=lora,
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

        image = self.encode_image(image, settings)
        self.load_encoded_image(image)

        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["point"]["prefix"]
                + self.tokenizer.encode(" " + object).ids
                + self.config.tokenizer.templates["point"]["suffix"]
            ],
            device=self.device,
        )

        lora = (
            variant_state_dict(settings["variant"], device=self.device)
            if settings is not None and "variant" in settings
            else None
        )

        _, hidden, next_token, pos = self._prefill_prompt(
            prompt_tokens, image.pos, temperature=0, top_p=0, lora=lora
        )
        hidden = hidden[:, -1:, :]

        max_objects = (
            settings.get("max_objects", DEFAULT_MAX_OBJECTS)
            if settings
            else DEFAULT_MAX_OBJECTS
        )
        objects = self._generate_points(
            hidden,
            next_token,
            pos,
            include_size=False,
            max_objects=max_objects,
            lora=lora,
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
                torch.tensor([[[source[0]]]], device=self.device, dtype=torch.bfloat16),
                self.region,
            )
            y_emb = encode_coordinate(
                torch.tensor([[[source[1]]]], device=self.device, dtype=torch.bfloat16),
                self.region,
            )

            prompt_emb = torch.cat([before_emb, x_emb, y_emb, after_emb], dim=1)

            self.load_encoded_image(image)

            mask = self.attn_mask[:, :, image.pos : image.pos + prompt_emb.size(1), :]
            pos_ids = torch.arange(
                image.pos, image.pos + prompt_emb.size(1), dtype=torch.long
            )
            hidden = self._prefill(prompt_emb, mask, pos_ids, lora=None)
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
