from dataclasses import dataclass


@dataclass(frozen=True)
class TextConfig:
    dim: int = 2048
    n_layers: int = 24
    vocab_size: int = 51200
    max_context: int = 2048
    n_heads: int = 32


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


@dataclass(frozen=True)
class RegionConfig:
    dim: int = 2048
    coord_feat_dim: int = 256
    coord_out_dim: int = 1024
    size_feat_dim: int = 512
    size_out_dim: int = 2048


@dataclass
class MoondreamConfig:
    text: TextConfig = TextConfig()
    vision: VisionConfig = VisionConfig()
    region: RegionConfig = RegionConfig()

    @classmethod
    def from_dict(cls, config_dict: dict):
        text_config = TextConfig(**config_dict.get("text", {}))
        vision_config = VisionConfig(**config_dict.get("vision", {}))
        region_config = RegionConfig(**config_dict.get("region", {}))
        return cls(text=text_config, vision=vision_config, region=region_config)
