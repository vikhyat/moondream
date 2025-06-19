from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TextConfig:
    dim: int = 2048
    ff_dim: int = 8192
    n_layers: int = 24
    vocab_size: int = 51200
    max_context: int = 2048
    n_heads: int = 32
    n_kv_heads: int = 32
    prefix_attn: int = 730
    group_size: Optional[int] = None


@dataclass(frozen=True)
class VisionConfig:
    enc_dim: int = 1152
    enc_patch_size: int = 14
    enc_n_layers: int = 27
    enc_ff_dim: int = 4304
    enc_n_heads: int = 16
    proj_out_dim: int = 2048
    crop_size: int = 378
    in_channels: int = 3
    max_crops: int = 12
    overlap_margin: int = 4
    proj_inner_dim: int = 8192


@dataclass(frozen=True)
class RegionConfig:
    dim: int = 2048
    coord_feat_dim: int = 256
    coord_out_dim: int = 1024
    size_feat_dim: int = 512
    size_out_dim: int = 2048
    inner_dim: int = 8192
    group_size: Optional[int] = None


@dataclass(frozen=True)
class TokenizerConfig:
    bos_id: int = 0
    eos_id: int = 0
    answer_id: int = 3
    thinking_id: int = 4
    coord_id: int = 5
    size_id: int = 6
    start_ground_points_id: int = 7
    end_ground_id: int = 9
    templates: Dict[str, Optional[Dict[str, List[int]]]] = field(
        default_factory=lambda: {
            "caption": {
                "short": [1, 32708, 2, 12492, 3],
                "normal": [1, 32708, 2, 6382, 3],
                "long": [1, 32708, 2, 4059, 3],
            },
            "query": {"prefix": [1, 15381, 2], "suffix": [3]},
            "detect": {"prefix": [1, 7235, 476, 2], "suffix": [3]},
            "point": {"prefix": [1, 2581, 2], "suffix": [3]},
        }
    )


@dataclass(frozen=True)
class MoondreamConfig:
    text: TextConfig = TextConfig()
    vision: VisionConfig = VisionConfig()
    region: RegionConfig = RegionConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()

    @classmethod
    def from_dict(cls, config_dict: dict):
        text_config = TextConfig(**config_dict.get("text", {}))
        vision_config = VisionConfig(**config_dict.get("vision", {}))
        region_config = RegionConfig(**config_dict.get("region", {}))
        tokenizer_config = TokenizerConfig(**config_dict.get("tokenizer", {}))
        return cls(
            text=text_config,
            vision=vision_config,
            region=region_config,
            tokenizer=tokenizer_config,
        )

    def to_dict(self):
        return {
            "text": self.text.__dict__,
            "vision": self.vision.__dict__,
            "region": self.region.__dict__,
            "tokenizer": self.tokenizer.__dict__,
        }
