import torch
import torch.nn as nn

from .config import MoondreamConfig


class MoondreamModel(nn.Module):
    def __init__(self, config: MoondreamConfig, dtype=torch.float16):
        super().__init__()
        self.config = config

        # Vision Model
        patch_dim = (
            config.vision.enc_patch_size
            * config.vision.enc_patch_size
            * config.vision.in_channels
        )
        grid_size = config.vision.crop_size // config.vision.enc_patch_size
        num_patches = grid_size * grid_size

        self.vision = nn.ModuleDict(
            {
                "patch_emb": nn.Linear(patch_dim, config.vision.enc_dim, dtype=dtype),
                "blocks": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "ln1": nn.LayerNorm(config.vision.enc_dim, dtype=dtype),
                                "attn": nn.ModuleDict(
                                    {
                                        "qkv": nn.Linear(
                                            config.vision.enc_dim,
                                            3 * config.vision.enc_dim,
                                            dtype=dtype,
                                        ),
                                        "proj": nn.Linear(
                                            config.vision.enc_dim,
                                            config.vision.enc_dim,
                                            dtype=dtype,
                                        ),
                                    }
                                ),
                                "ln2": nn.LayerNorm(config.vision.enc_dim, dtype=dtype),
                                "mlp": nn.ModuleDict(
                                    {
                                        "fc1": nn.Linear(
                                            config.vision.enc_dim,
                                            config.vision.enc_ff_dim,
                                            dtype=dtype,
                                        ),
                                        "fc2": nn.Linear(
                                            config.vision.enc_ff_dim,
                                            config.vision.enc_dim,
                                            dtype=dtype,
                                        ),
                                    }
                                ),
                            }
                        )
                        for _ in range(config.vision.enc_n_layers)
                    ]
                ),
                "post_ln": nn.LayerNorm(config.vision.enc_dim, dtype=dtype),
                "proj_mlp": nn.ModuleDict(
                    {
                        "fc1": nn.Linear(
                            config.vision.enc_dim * 2, config.text.dim * 4, dtype=dtype
                        ),
                        "fc2": nn.Linear(
                            config.text.dim * 4, config.text.dim, dtype=dtype
                        ),
                    }
                ),
            }
        )
        self.vision.pos_emb = nn.Parameter(
            torch.zeros(1, num_patches, config.vision.enc_dim, dtype=dtype)
        )

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
