# Ethically sourced from https://github.com/xjdr-alt/entropix

import torch
from typing import Tuple


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=dtype).unsqueeze(1)
    freqs = t * freqs.unsqueeze(0)
    freqs = torch.exp(1j * freqs)
    return torch.stack([freqs.real, freqs.imag], dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if interleave:
        xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 0]
        xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 1]
        xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 0]
        xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 1]
    else:
        d_q, d_k = xq.shape[-1] // 2, xk.shape[-1] // 2
        xq_r, xq_i = xq[..., :d_q], xq[..., d_q:]
        xk_r, xk_i = xk[..., :d_k], xk[..., d_k:]

    freqs_cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(0)

    # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)
    xk_out = torch.stack((xk_out_r, xk_out_i), dim=-1).flatten(-2)

    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)
