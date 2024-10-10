import torch
import math
from typing import Literal
from torch.nn import functional as F
from dataclasses import dataclass


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
    # return F.layer_norm(x, (1152,), w.weight, w.bias)


@dataclass
class MLPWeights:
    fc1: LinearWeights
    fc2: LinearWeights
    act: Literal["gelu_approx", "gelu_new"] = "gelu_approx"


def mlp(x: torch.Tensor, w: MLPWeights) -> torch.Tensor:
    x = linear(x, w.fc1)
    if w.act == "gelu_approx":
        x = F.gelu(x, approximate="tanh")
    elif w.act == "gelu_new":
        x = (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )
    else:
        raise NotImplementedError(f"Activation function {w.act} not implemented.")
    x = linear(x, w.fc2)
    return x


@dataclass
class AttentionWeights:
    qkv: LinearWeights
    proj: LinearWeights
    n_heads: int
