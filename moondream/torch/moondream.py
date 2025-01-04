import torch
import torch.nn as nn

from typing import Union
from PIL import Image
from dataclasses import dataclass
from tokenizers import Tokenizer

from .config import MoondreamConfig
from .image_crops import reconstruct_from_crops
from .vision import vision_encoder, vision_projection, prepare_crops, build_vision_model
from .text import build_text_model, prefill, text_encoder, lm_head, decode_one_token


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
        self.ops["vision_projection"] = torch.compile(
            self.ops["vision_projection"], fullgraph=True
        )
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

    def encode_image(self, image: Image.Image) -> EncodedImage:
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

    def query(
        self,
        image: Union[Image.Image, EncodedImage],
        question: str,
        stream: bool = False,
    ):
        if isinstance(image, Image.Image):
            image = self.encode_image(image)
        elif not isinstance(image, EncodedImage):
            raise ValueError("image must be a PIL Image or EncodedImage")

        if self.config.tokenizer.templates["query"] is None:
            raise NotImplementedError("query not supported for this model")

        question_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["query"]["prefix"]
                + self.tokenizer.encode(question).ids
                + self.config.tokenizer.templates["query"]["suffix"]
            ],
            device=self.device,
        )

        # Prefill with the question.
        kv_cache = image.kv_cache.clone()
        with torch.no_grad():
            question_emb = text_encoder(question_tokens, self.text)
            hidden = self.ops["prefill"](
                question_emb, kv_cache, image.pos, self.text, self.config.text
            )
            logits = lm_head(hidden, self.text)
            next_token = torch.argmax(logits, dim=-1)
        pos = image.pos + question_emb.size(1)

        # Decode logits one by one.
        def generator(next_token, pos):
            while (next_token_id := next_token.item()) != self.config.tokenizer.eos_id:
                yield self.tokenizer.decode([next_token_id])

                with torch.no_grad():
                    logits, hidden, kv_cache_update = self.ops["decode_one_token"](
                        next_token, kv_cache, pos, self.text, self.config.text
                    )
                    kv_cache[:, :, :, :, pos : pos + kv_cache_update.size(-2), :] = (
                        kv_cache_update
                    )
                    pos += 1
                    next_token = torch.argmax(logits, dim=-1)

        if stream:
            return {"answer": generator(next_token, pos)}
        else:
            return {"answer": "".join(list(generator(next_token, pos)))}
