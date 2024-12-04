# Ethically sourced from https://github.com/xjdr-alt/entropix

import torch


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
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int,
    rot_dim: int = 32,
    interleave: bool = False,
) -> torch.Tensor:
    assert rot_dim == freqs_cis.shape[-2] * 2
    assert num_heads == x.shape[1]

    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    if interleave:
        xq_r = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 0]
        xq_i = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 1]
    else:
        d_q = x_rot.shape[-1] // 2
        xq_r, xq_i = x_rot[..., :d_q], x_rot[..., d_q:]

    freqs_cos = freqs_cis[..., 0][position_ids, :].unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_cis[..., 1][position_ids, :].unsqueeze(0).unsqueeze(0)

    # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)

    return torch.cat([xq_out.to(x.dtype), x_pass], dim=-1)
