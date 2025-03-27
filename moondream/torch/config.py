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


@dataclass(frozen=True)
class TokenizerConfig:
    bos_id: int = 50256
    eos_id: int = 50256
    templates: Dict[str, Optional[Dict[str, List[int]]]] = field(
        default_factory=lambda: {
            "caption": {
                "short": [198, 198, 16438, 8305, 25],
                "normal": [198, 198, 24334, 1159, 25],
                "long": [198, 198, 14617, 8305, 25],
            },
            "query": {"prefix": [198, 198, 24361, 25], "suffix": [198, 198, 33706, 25]},
            "detect": {"prefix": [198, 198, 47504, 25], "suffix": [628]},
            "point": {"prefix": [198, 198, 12727, 25], "suffix": [628]},
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
