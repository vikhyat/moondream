from transformers import PretrainedConfig

from typing import Optional
import math


class PhiConfig(PretrainedConfig):
    model_type = "phi-msft"

    def __init__(
        self,
        vocab_size: int = 51200,
        n_positions: int = 2048,
        n_embd: int = 2048,
        n_layer: int = 24,
        n_inner: Optional[int] = None,
        n_head: int = 32,
        n_head_kv: Optional[int] = None,
        rotary_dim: Optional[int] = 32,
        activation_function: Optional[str] = "gelu_new",
        flash_attn: bool = False,
        flash_rotary: bool = False,
        fused_dense: bool = False,
        attn_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        pad_vocab_size_multiple: int = 64,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        pad_vocab_size = (
            math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        )
        super().__init__(
            vocab_size=pad_vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_inner=n_inner,
            n_head=n_head,
            n_head_kv=n_head_kv,
            activation_function=activation_function,
            attn_pdrop=attn_pdrop,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            pad_vocab_size_multiple=pad_vocab_size_multiple,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs
        )
        self.rotary_dim = min(rotary_dim, n_embd // n_head)
        self.flash_attn = flash_attn
        self.flash_rotary = flash_rotary
        self.fused_dense = fused_dense

    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }


class MoondreamConfig(PretrainedConfig):
    model_type = "moondream1"

    def __init__(self, **kwargs):
        self.phi_config = PhiConfig(**kwargs)
        super().__init__(**kwargs)
