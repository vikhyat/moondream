import torch
import torch.nn as nn

from PIL import Image

from .config import MoondreamConfig
from .vision import vision_encoder, vision_projection, prepare_crops, build_vision_model
from .image_crops import reconstruct_from_crops


class MoondreamModel(nn.Module):
    def __init__(self, config: MoondreamConfig, dtype=torch.float16):
        super().__init__()
        self.config = config

        self.vision = build_vision_model(config.vision, dtype)

        # Text Model
        self.text = nn.ModuleDict(
            {
                "blocks": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "ln": nn.LayerNorm(config.text.dim, dtype=dtype),
                                "attn": nn.ModuleDict(
                                    {
                                        "qkv": nn.Linear(
                                            config.text.dim,
                                            3 * config.text.dim,
                                            dtype=dtype,
                                        ),
                                        "proj": nn.Linear(
                                            config.text.dim,
                                            config.text.dim,
                                            dtype=dtype,
                                        ),
                                    }
                                ),
                                "mlp": nn.ModuleDict(
                                    {
                                        "fc1": nn.Linear(
                                            config.text.dim,
                                            4 * config.text.dim,
                                            dtype=dtype,
                                        ),
                                        "fc2": nn.Linear(
                                            4 * config.text.dim,
                                            config.text.dim,
                                            dtype=dtype,
                                        ),
                                    }
                                ),
                            }
                        )
                        for _ in range(config.text.n_layers)
                    ]
                ),
                "post_ln": nn.LayerNorm(config.text.dim, dtype=dtype),
                "lm_head": nn.Linear(
                    config.text.dim, config.text.vocab_size, dtype=dtype
                ),
            }
        )
        self.text.wte = nn.Parameter(
            torch.empty(config.text.vocab_size, config.text.dim, dtype=dtype)
        )

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
        }

    @property
    def device(self):
        return self.vision.pos_emb.device

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        all_crops, tiling = prepare_crops(image, self.config.vision, device=self.device)

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
