import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import layer_norm, linear, mlp
from .rope import apply_rotary_emb, precompute_freqs_cis
from .weights import AttentionWeights
from .config import TextConfig


def text_encoder(input_ids: torch.Tensor, w: nn.Module):
    return F.embedding(input_ids, w.wte)


def attn_mask(pos, seq_len):
    """
    Create an attention mask that aligns with the bottom right of the
    attention matrix. For example, if q_len = 2 and kv_len = 5, we want the
    following:

         1 1 1 1 0
         1 1 1 1 1

    and not this, which is what we get by default if we just set is_causal.

         1 0 0 0 0
         1 1 0 0 0
    """
    mask = torch.ones(seq_len, pos + seq_len, dtype=torch.bool)
    mask[:, pos:] = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    return mask


def attn(
    x: torch.Tensor,
    w: AttentionWeights,
    freqs_cis: torch.Tensor,
    layer_kv_cache: torch.Tensor,
    n_heads: int,
    pos: int,
):
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    q, k, v = [
        t.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        for t in linear(x, w.qkv).chunk(3, dim=-1)
    ]

    position_ids = torch.arange(pos, pos + q_len, dtype=torch.long)
    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_heads)

    k_, v_ = k, v
    if layer_kv_cache is not None:
        k = torch.cat([layer_kv_cache[0, :, :, :pos, :], k], dim=2)
        v = torch.cat([layer_kv_cache[1, :, :, :pos, :], v], dim=2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask(pos, q_len)).to(
        # This type conversion isn't needed when running in PyTorch directly, but the
        # ONNX export runs attention in float32 because the attention mask is cast to
        # float32.
        x.dtype
    )
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out = linear(out, w.proj)
    return out, torch.stack([k_, v_])


def text_decoder(
    inputs_embeds: torch.Tensor,
    w: nn.Module,
    kv_cache: torch.Tensor,
    pos: int,
    config: TextConfig,
):
    hidden_BTC = inputs_embeds
    new_kv_cache = [torch.empty(0)] * len(w.blocks)

    for i, block in enumerate(w.blocks):
        l_in = layer_norm(hidden_BTC, block.ln)
        l_attn, new_kv_cache[i] = attn(
            l_in, block.attn, w.freqs_cis, kv_cache[i], n_heads=config.n_heads, pos=pos
        )
        l_mlp = mlp(l_in, block.mlp)
        hidden_BTC = hidden_BTC + l_attn + l_mlp

    return hidden_BTC, torch.stack(new_kv_cache)


def lm_head(hidden_BTC: torch.Tensor, w: nn.Module):
    hidden_BC = hidden_BTC[:, -1, :]
    hidden_BC = layer_norm(hidden_BC, w.post_ln)
    logits = linear(hidden_BC, w.lm_head)
    return logits


def prefill(
    inputs_embeds: torch.Tensor,
    kv_cache: torch.Tensor,
    pos: int,
    w: nn.Module,
    config: TextConfig,
):
    # Updates kv_cache in-place
    hidden, kv_cache[:, :, :, :, pos : pos + inputs_embeds.size(1), :] = text_decoder(
        inputs_embeds, w, kv_cache[:, :, :, :, :pos, :], pos, config
    )
    return hidden


def decode_one_token(
    last_token: torch.Tensor,
    kv_cache: torch.Tensor,
    pos: int,
    w: nn.Module,
    config: TextConfig,
):
    token_emb = text_encoder(last_token[None], w)
    hidden, kv_cache_update = text_decoder(token_emb, w, kv_cache, pos, config)
    logits = lm_head(hidden, w)
    return logits, hidden, kv_cache_update


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
