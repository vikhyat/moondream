import torch
import torch.nn as nn

from torch.nn import functional as F

from .layers import layer_norm, linear, mlp
from .rope import apply_rotary_emb, precompute_freqs_cis
from .config import TextConfig


def text_encoder(input_ids: torch.Tensor, w: nn.Module):
    return F.embedding(input_ids, w.wte)


def attn(
    x: torch.Tensor,
    w: nn.Module,
    freqs_cis: torch.Tensor,
    kv_cache: nn.Module,
    attn_mask: torch.Tensor,
    n_heads: int,
    position_ids: torch.Tensor,
):
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    q, k, v = [
        t.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        for t in linear(x, w.qkv).chunk(3, dim=-1)
    ]

    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_heads)

    if kv_cache is not None:
        k, v = kv_cache.update(position_ids, k, v)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).to(
        # This type conversion isn't needed when running in PyTorch directly, but the
        # ONNX export runs attention in float32 because the attention mask is cast to
        # float32.
        x.dtype
    )
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out = linear(out, w.proj)
    return out


def text_decoder(
    x: torch.Tensor,
    w: nn.Module,
    attn_mask: torch.Tensor,
    position_ids: torch.Tensor,
    config: TextConfig,
):
    for i, block in enumerate(w.blocks):
        l_in = layer_norm(x, block.ln)
        l_attn = attn(
            l_in,
            block.attn,
            freqs_cis=w.freqs_cis,
            kv_cache=block.kv_cache,
            attn_mask=attn_mask,
            n_heads=config.n_heads,
            position_ids=position_ids,
        )
        l_mlp = mlp(l_in, block.mlp)
        x = x + l_attn + l_mlp

    return x


def lm_head(hidden_BTC: torch.Tensor, w: nn.Module):
    hidden_BC = hidden_BTC[:, -1, :]
    hidden_BC = layer_norm(hidden_BC, w.post_ln)
    logits = linear(hidden_BC, w.lm_head)
    return logits


def prefill(
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    pos_ids: torch.Tensor,
    w: nn.Module,
    config: TextConfig,
):
    hidden = text_decoder(x, w, attn_mask, pos_ids, config)
    return hidden


def decode_one_token(
    x: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_mask: torch.Tensor,
    pos_ids: torch.Tensor,
    w: nn.Module,
    config: TextConfig,
):
    hidden = text_decoder(x[None], w, attn_mask, pos_ids, config)
    logits = lm_head(hidden, w)
    return logits, hidden


def build_text_model(config: TextConfig, dtype: torch.dtype) -> nn.Module:
    text = nn.ModuleDict(
        {
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln": nn.LayerNorm(config.dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": nn.Linear(
                                        config.dim, 3 * config.dim, dtype=dtype
                                    ),
                                    "proj": nn.Linear(
                                        config.dim, config.dim, dtype=dtype
                                    ),
                                }
                            ),
                            "mlp": nn.ModuleDict(
                                {
                                    "fc1": nn.Linear(
                                        config.dim, 4 * config.dim, dtype=dtype
                                    ),
                                    "fc2": nn.Linear(
                                        4 * config.dim, config.dim, dtype=dtype
                                    ),
                                }
                            ),
                        }
                    )
                    for _ in range(config.n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.dim, dtype=dtype),
            "lm_head": nn.Linear(config.dim, config.vocab_size, dtype=dtype),
        }
    )
    text.wte = nn.Parameter(torch.empty(config.vocab_size, config.dim, dtype=dtype))
    text.register_buffer(
        "freqs_cis",
        precompute_freqs_cis(config.dim // (2 * config.n_heads), config.max_context),
        persistent=False,
    )

    return text
