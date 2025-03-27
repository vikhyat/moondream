from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F


def gelu_approx(x):
    return F.gelu(x, approximate="tanh")


@dataclass
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def linear(x: torch.Tensor, w: LinearWeights) -> torch.Tensor:
    return F.linear(x, w.weight, w.bias)


@dataclass
class LayerNormWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def layer_norm(x: torch.Tensor, w: LayerNormWeights) -> torch.Tensor:
    return F.layer_norm(x, w.bias.shape, w.weight, w.bias)


@dataclass
class MLPWeights:
    fc1: LinearWeights
    fc2: LinearWeights
    act: Literal["gelu_approx"] = "gelu_approx"


def mlp(x: torch.Tensor, w: MLPWeights) -> torch.Tensor:
    x = w.fc1(x)
    x = gelu_approx(x)
    x = w.fc2(x)
    return x


@dataclass
class AttentionWeights:
    qkv: LinearWeights
    proj: LinearWeights


def attn(x: torch.Tensor, w: AttentionWeights, n_heads: int) -> torch.Tensor:
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    q, k, v = [
        t.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        for t in linear(x, w.qkv).chunk(3, dim=-1)
    ]
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out = linear(out, w.proj)
    return out
