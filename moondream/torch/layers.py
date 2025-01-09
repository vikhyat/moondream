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


@torch.compile
def fast_int8_lin(x, w, scale, bias):
    mm = F.linear(x.to(torch.float16), w.to(torch.float16))
    mm = mm * scale
    mm = mm + bias
    return mm


def quantize_x(x: torch.Tensor) -> torch.Tensor:
    x_min = -F.relu(-x.min())
    x_max = F.relu(x.max())
    x_scale = (x_max - x_min) / 255.0
    x_zero_point = torch.round(torch.clamp(-x_min / x_scale, 0, 255))
    x_quant = torch.clamp(
        torch.round((x - x_zero_point * x_scale) / x_scale), -128, 127
    ).to(torch.int8)
    return x_quant, x_scale

def linear(x: torch.Tensor, w: LinearWeights) -> torch.Tensor:
    if hasattr(w, "w_scale") is not None:
        x_quant, x_scale = quantize_x(x)
        fast_int8_lin(x_quant.to(torch.int8), w.weight.to(torch.int8), x_scale, w.bias)
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
    x = linear(x, w.fc1)
    x = gelu_approx(x)
    x = linear(x, w.fc2)
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
