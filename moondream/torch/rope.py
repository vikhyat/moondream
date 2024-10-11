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

    t = torch.arange(end, dtype=dtype).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if interleave:
        reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
        reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
        xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
        xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    else:
        d_q, d_k = xq.shape[-1] // 2, xk.shape[-1] // 2
        xq_ = torch.complex(xq[..., :d_q], xq[..., d_q:])
        xk_ = torch.complex(xk[..., :d_k], xk[..., d_k:])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(0)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(0)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(
        *xq_out.shape[:-1], -1
    )
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(
        *xk_out.shape[:-1], -1
    )
    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)
