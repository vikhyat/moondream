import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Literal
from torchao import quantize_
from torchao.quantization import int4_weight_only

from .packing import dequantize_tensor


def gelu_approx(x):
    return F.gelu(x, approximate="tanh")


@dataclass
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def linear(x: torch.Tensor, w: LinearWeights) -> torch.Tensor:
    return F.linear(x, w.weight, w.bias)


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
    ):
        # TODO: Take group_size as an input instead of hardcoding it here.
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.ParameterDict(
            {
                "packed": nn.Parameter(
                    torch.empty(
                        out_features, in_features // 128, 64, dtype=torch.uint8
                    ),
                    requires_grad=False,
                ),
                "scales": nn.Parameter(
                    torch.empty(out_features, in_features // 128), requires_grad=False
                ),
            }
        )
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        self.unpacked = False

    def unpack(self):
        self.weight = nn.Parameter(
            dequantize_tensor(
                self.weight["packed"],
                self.weight["scales"],
                (self.weight["packed"].shape[0], self.weight["packed"].shape[1] * 128),
                128,
                torch.bfloat16,
            )
        )
        with torch.device("meta"):
            self.linear = nn.Linear(
                self.in_features, self.out_features, dtype=torch.bfloat16
            )
        self.linear.weight = self.weight
        self.linear.bias = nn.Parameter(
            self.bias.to(torch.bfloat16), requires_grad=False
        )
        del self.weight, self.bias
        quantize_(self, int4_weight_only(group_size=128))
        torch.cuda.empty_cache()
        self.unpacked = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.unpacked:
            self.unpack()
        return self.linear(x)


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
