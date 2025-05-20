from dataclasses import dataclass
from typing import Literal

import bitblas
from bitblas.cache import OperatorCache

import torch
from torch.nn import functional as F
import torch.nn as nn


def gelu_approx(x):
    return F.gelu(x, approximate="tanh")


@dataclass
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor


class Linear(nn.Module):
    """
    Linear layer with support for bitblas quantization.
    If dtype is torch.int8, it uses bitblas for quantization.
    Otherwise, it uses a standard nn.Linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        operator_cache: OperatorCache = None,
        cache_dir: str = None,
        group_size: int = 128,
    ):
        super().__init__()

        if dtype == torch.int8:
            self.linear = bitblas.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                with_zeros=True,
                zeros_mode="original",
                with_scaling=True,
                A_dtype="float16",
                W_dtype="uint4",
                accum_dtype="float16",
                out_dtype="float16",
                fast_decoding=True,
                enable_tuning=True,
                operator_cache=operator_cache,
                database_path=cache_dir,
                group_size=group_size,
            )
        else:
            self.linear = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                dtype=torch.float16,
            )

    def forward(self, x):
        return self.linear(x)

    @property
    def weight(self) -> torch.Tensor:
        try:
            return self.linear.weight
        except AttributeError:
            return self.linear.qweight

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


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
