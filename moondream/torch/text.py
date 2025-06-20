import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Optional

from .layers import layer_norm, mlp, QuantizedLinear
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
    n_kv_heads: int,
    position_ids: torch.Tensor,
    lora: Optional[dict],
):
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    qkv_out = w.qkv(x)  # shape: (bsz, q_len, (n_heads + 2*n_kv_heads)*head_dim)
    if lora is not None:
        qkv_out += F.linear(F.linear(x, lora["qkv"]["A"]), lora["qkv"]["B"])
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)
    del qkv_out

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_kv_heads)

    if kv_cache is not None:
        k, v = kv_cache.update(position_ids, k, v)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, enable_gqa=n_heads != n_kv_heads
    )
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)

    out0 = w.proj(out)
    if lora is not None:
        out1 = F.linear(F.linear(x, lora["proj"]["A"]), lora["proj"]["B"])
        out = out0 + out1
    else:
        out = out0

    return out


def _attn(
    x: torch.Tensor,
    w: torch.Tensor,
    freqs_cis: torch.Tensor,
    attn_mask: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
):
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads
    pos = 0

    qkv_out = w.qkv(x)  # shape: (bsz, q_len, (n_heads + 2*n_kv_heads)*head_dim)
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    q = qkv_out[..., :q_dim].view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = (
        qkv_out[..., q_dim : q_dim + kv_dim]
        .view(bsz, q_len, n_kv_heads, head_dim)
        .transpose(1, 2)
    )
    v = (
        qkv_out[..., q_dim + kv_dim :]
        .view(bsz, q_len, n_kv_heads, head_dim)
        .transpose(1, 2)
    )

    position_ids = torch.arange(pos, pos + q_len, dtype=torch.long)
    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_kv_heads)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, enable_gqa=n_heads != n_kv_heads
    )
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out = w.proj(out)
    return out


def _produce_hidden(inputs_embeds: torch.Tensor, w: nn.Module, config: TextConfig):
    hidden_BTC = inputs_embeds

    bsz, q_len, d_model = inputs_embeds.shape
    attn_mask = torch.zeros(q_len, q_len)
    attn_mask[:730, :730] = 1
    for i in range(730, q_len):
        attn_mask[i, : i + 1] = 1
    attn_mask = attn_mask.to(dtype=torch.bool)

    for i, block in enumerate(w.blocks):
        l_in = layer_norm(hidden_BTC, block.ln)
        l_attn = _attn(
            x=l_in,
            w=block.attn,
            freqs_cis=w.freqs_cis,
            attn_mask=attn_mask,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
        )
        l_mlp = mlp(l_in, block.mlp)
        hidden_BTC = hidden_BTC + l_attn + l_mlp

    return hidden_BTC


def text_decoder(
    x: torch.Tensor,
    w: nn.Module,
    attn_mask: torch.Tensor,
    position_ids: torch.Tensor,
    config: TextConfig,
    lora: Optional[dict],
):
    for i, block in enumerate(w.blocks):
        if lora is not None:
            layer_lora = lora["text"]["blocks"][str(i)]
            mlp_lora = layer_lora["mlp"]
            attn_lora = layer_lora["attn"]
        else:
            mlp_lora = None
            attn_lora = None

        l_in = layer_norm(x, block.ln)
        l_attn = attn(
            l_in,
            block.attn,
            freqs_cis=w.freqs_cis,
            kv_cache=block.kv_cache,
            attn_mask=attn_mask,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            position_ids=position_ids,
            lora=attn_lora,
        )
        l_mlp = mlp(l_in, block.mlp, lora=mlp_lora)
        x = x + l_attn + l_mlp

    return x


def lm_head(hidden_BTC: torch.Tensor, w: nn.Module):
    hidden_BC = hidden_BTC[:, -1, :]
    hidden_BC = layer_norm(hidden_BC, w.post_ln)
    logits = w.lm_head(hidden_BC)
    return logits


def _lm_head(hidden_BTC: torch.Tensor, w: nn.Module):
    hidden_BTC = layer_norm(hidden_BTC, w.post_ln)
    logits = w.lm_head(hidden_BTC)
    return logits


def build_text_model(config: TextConfig, dtype: torch.dtype) -> nn.Module:
    qkv_dim = int(config.dim * (1 + 2 * config.n_kv_heads / config.n_heads))
    linear_cls = QuantizedLinear if config.group_size is not None else nn.Linear

    text = nn.ModuleDict(
        {
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln": nn.LayerNorm(config.dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": linear_cls(config.dim, qkv_dim, dtype=dtype),
                                    "proj": linear_cls(
                                        config.dim, config.dim, dtype=dtype
                                    ),
                                }
                            ),
                            "mlp": nn.ModuleDict(
                                {
                                    "fc1": linear_cls(
                                        config.dim, config.ff_dim, dtype=dtype
                                    ),
                                    "fc2": linear_cls(
                                        config.ff_dim, config.dim, dtype=dtype
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
