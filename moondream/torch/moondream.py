import torch
import torch.nn as nn

from typing import Literal, Union
from PIL import Image
from dataclasses import dataclass
from tokenizers import Tokenizer

from .config import MoondreamConfig
from .image_crops import reconstruct_from_crops
from .vision import vision_encoder, vision_projection, prepare_crops, build_vision_model
from .text import build_text_model, prefill, text_encoder, lm_head, decode_one_token
from .region import decode_coordinate, encode_coordinate, decode_size, encode_size


@dataclass(frozen=True)
class EncodedImage:
    pos: int
    kv_cache: torch.Tensor


class MoondreamModel(nn.Module):
    def __init__(self, config: MoondreamConfig, dtype=torch.float16):
        super().__init__()
        self.config = config

        self.tokenizer = Tokenizer.from_pretrained("vikhyatk/moondream-next")
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
                            config.region.dim, config.region.dim * 4, dtype=dtype
                        ),
                        "fc2": nn.Linear(
                            config.region.dim * 4,
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
                            config.region.dim, config.region.dim * 4, dtype=dtype
                        ),
                        "fc2": nn.Linear(
                            config.region.dim * 4,
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

        self.ops = {
            "vision_encoder": vision_encoder,
            "vision_projection": vision_projection,
            "prefill": prefill,
            "decode_one_token": decode_one_token,
        }

    @property
    def device(self):
        return self.vision.pos_emb.device

    def compile(self):
        self.ops["vision_encoder"] = torch.compile(
            self.ops["vision_encoder"], fullgraph=True
        )
        # Need to figure out how to mark the 'reconstructed' input shape as dynamic
        # self.ops["vision_projection"] = torch.compile(
        #     self.ops["vision_projection"], fullgraph=True
        # )
        self.ops["prefill"] = torch.compile(self.ops["prefill"], fullgraph=True)
        self.ops["decode_one_token"] = torch.compile(
            self.ops["decode_one_token"], fullgraph=True
        )

    def _run_vision_encoder(self, image: Image.Image) -> torch.Tensor:
        all_crops, tiling = prepare_crops(image, self.config.vision, device=self.device)
        torch._dynamo.mark_dynamic(all_crops, 0)

        outputs = self.ops["vision_encoder"](all_crops, self.vision, self.config.vision)

        global_features = outputs[0]
        local_features = outputs[1:].view(-1, 27, 27, 1152)

        reconstructed = reconstruct_from_crops(
            local_features,
            tiling,
            patch_size=1,
            overlap_margin=self.config.vision.overlap_margin,
        )

        return self.ops["vision_projection"](
            global_features, reconstructed, self.vision
        )

    def encode_image(self, image: Union[Image.Image, EncodedImage]) -> EncodedImage:
        if isinstance(image, EncodedImage):
            return image
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image or EncodedImage")

        # Run through text model in addition to the vision encoder, to minimize
        # re-computation if multiple queries are performed on this image.
        kv_cache = torch.zeros(
            self.config.text.n_layers,
            2,  # k, v
            1,  # batch size
            self.config.text.n_heads,
            self.config.text.max_context,  # static cache
            self.config.text.dim // self.config.text.n_heads,  # head dim
            device=self.device,
            dtype=torch.float16,
        )
        with torch.no_grad():
            img_emb = self._run_vision_encoder(image)
            bos_emb = text_encoder(
                torch.tensor([[self.config.tokenizer.bos_id]], device=self.device),
                self.text,
            )
            inputs_embeds = torch.cat([bos_emb, img_emb[None]], dim=1)
            self.ops["prefill"](inputs_embeds, kv_cache, 0, self.text, self.config.text)
        return EncodedImage(pos=inputs_embeds.size(1), kv_cache=kv_cache)

    def _prefill_prompt(
        self, kv_cache: torch.Tensor, prompt_tokens: torch.Tensor, pos: int
    ):
        with torch.no_grad():
            prompt_emb = text_encoder(prompt_tokens, self.text)
            hidden = self.ops["prefill"](
                prompt_emb, kv_cache, pos, self.text, self.config.text
            )
            logits = lm_head(hidden, self.text)
            next_token = torch.argmax(logits, dim=-1)
        pos = pos + prompt_emb.size(1)
        return logits, hidden, next_token, pos

    def _generate_text(
        self, prompt_tokens: torch.Tensor, kv_cache: torch.Tensor, pos: int
    ):
        kv_cache = kv_cache.clone()
        _, _, next_token, pos = self._prefill_prompt(kv_cache, prompt_tokens, pos)

        def generator(next_token, pos):
            while (
                next_token_id := next_token.item()
            ) != self.config.tokenizer.eos_id and pos < self.config.text.max_context:
                yield self.tokenizer.decode([next_token_id])

                with torch.no_grad():
                    next_emb = text_encoder(next_token, self.text)
                    logits, _, kv_cache_update = self.ops["decode_one_token"](
                        next_emb, kv_cache, pos, self.text, self.config.text
                    )
                    kv_cache[:, :, :, :, pos : pos + kv_cache_update.size(-2), :] = (
                        kv_cache_update
                    )
                    pos += 1
                    next_token = torch.argmax(logits, dim=-1)

        return generator(next_token, pos)

    def query(
        self,
        image: Union[Image.Image, EncodedImage],
        question: str,
        stream: bool = False,
    ):
        if self.config.tokenizer.templates["query"] is None:
            raise NotImplementedError("Model does not support querying.")

        image = self.encode_image(image)
        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["query"]["prefix"]
                + self.tokenizer.encode(question).ids
                + self.config.tokenizer.templates["query"]["suffix"]
            ],
            device=self.device,
        )

        def generator():
            for token in self._generate_text(prompt_tokens, image.kv_cache, image.pos):
                yield token

        if stream:
            return {"answer": generator()}
        else:
            return {"answer": "".join(list(generator()))}

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: Literal["normal", "short"] = "normal",
        stream: bool = False,
    ):
        if self.config.tokenizer.templates["caption"] is None:
            raise NotImplementedError("Model does not support captioning.")
        if length not in self.config.tokenizer.templates["caption"]:
            raise ValueError(f"Model does not support caption length '{length}'.")

        image = self.encode_image(image)
        prompt_tokens = torch.tensor(
            [self.config.tokenizer.templates["caption"][length]], device=self.device
        )

        def generator():
            for token in self._generate_text(prompt_tokens, image.kv_cache, image.pos):
                yield token

        if stream:
            return {"caption": generator()}
        else:
            return {"caption": "".join(list(generator()))}

    def _generate_points(
        self,
        hidden: torch.Tensor,
        kv_cache: torch.Tensor,
        next_token: torch.Tensor,
        pos: int,
        include_size: bool = True,
        max_points: int = 50,
    ):
        out = []

        with torch.no_grad():
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
                _, hidden, kv_cache_update = self.ops["decode_one_token"](
                    next_emb, kv_cache, pos, self.text, self.config.text
                )
                kv_cache[:, :, :, :, pos : pos + kv_cache_update.size(-2), :] = (
                    kv_cache_update
                )
                pos += 1
                y_logits = decode_coordinate(hidden, self.region)
                y_center = torch.argmax(y_logits, dim=-1) / y_logits.size(-1)
                next_emb = encode_coordinate(
                    y_center.to(dtype=y_logits.dtype), self.region
                )

                # Decode size
                if include_size:
                    logits, hidden, kv_cache_update = self.ops["decode_one_token"](
                        next_emb, kv_cache, pos, self.text, self.config.text
                    )
                    kv_cache[:, :, :, :, pos : pos + kv_cache_update.size(-2), :] = (
                        kv_cache_update
                    )
                    pos += 1
                    size_logits = decode_size(hidden, self.region)
                    w = torch.argmax(size_logits[0], dim=-1) / size_logits.size(-1)
                    h = torch.argmax(size_logits[1], dim=-1) / size_logits.size(-1)
                    next_emb = encode_size(
                        torch.tensor([w, h], dtype=size_logits.dtype), self.region
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
                logits, hidden, kv_cache_update = self.ops["decode_one_token"](
                    next_emb, kv_cache, pos, self.text, self.config.text
                )
                kv_cache[:, :, :, :, pos : pos + kv_cache_update.size(-2), :] = (
                    kv_cache_update
                )
                pos += 1
                next_token = torch.argmax(logits, dim=-1)

        return out

    def detect(self, image: Union[Image.Image, EncodedImage], object: str):
        if self.config.tokenizer.templates["detect"] is None:
            raise NotImplementedError("Model does not support object detection.")

        image = self.encode_image(image)
        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["detect"]["prefix"]
                + self.tokenizer.encode(object).ids
                + self.config.tokenizer.templates["detect"]["suffix"]
            ]
        )

        kv_cache = image.kv_cache.clone()
        _, hidden, next_token, pos = self._prefill_prompt(
            kv_cache, prompt_tokens, image.pos
        )
        hidden = hidden[:, -1:, :]

        objects = self._generate_points(
            hidden, kv_cache, next_token, pos, include_size=True, max_points=50
        )

        return {"objects": objects}

    def point(self, image: Union[Image.Image, EncodedImage], object: str):
        if self.config.tokenizer.templates["point"] is None:
            raise NotImplementedError("Model does not support pointing.")

        image = self.encode_image(image)
        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["point"]["prefix"]
                + self.tokenizer.encode(object).ids
                + self.config.tokenizer.templates["point"]["suffix"]
            ]
        )

        kv_cache = image.kv_cache.clone()
        _, hidden, next_token, pos = self._prefill_prompt(
            kv_cache, prompt_tokens, image.pos
        )
        hidden = hidden[:, -1:, :]

        objects = self._generate_points(
            hidden, kv_cache, next_token, pos, include_size=False, max_points=50
        )

        return {"points": objects}
