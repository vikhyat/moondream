import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Literal, Optional

try:
    from torchao import quantize_
    from torchao.quantization import int4_weight_only
except ImportError:

    def quantize_(model, quant_mode):
        raise ImportError(
            "torchao is not installed. Please install it with `pip install torchao`."
        )

    def int4_weight_only(group_size):
        raise ImportError(
            "torchao is not installed. Please install it with `pip install torchao`."
        )


def gelu_approx(x):
    return F.gelu(x, approximate="tanh")


@dataclass
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def linear(x: torch.Tensor, w: LinearWeights) -> torch.Tensor:
    return F.linear(x, w.weight, w.bias)


def dequantize_tensor(W_q, scale, zero, orig_shape, dtype=torch.bfloat16):
    _step = W_q.shape[0]
    W_r = torch.empty([2 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)
    W_r[:_step] = (W_q & 0b11110000) >> 4
    W_r[_step:] = W_q & 0b00001111
    W_r.sub_(zero).mul_(scale)
    return W_r.reshape(orig_shape)


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
                        out_features * in_features // (128 * 2), 128, dtype=torch.uint8
                    ),
                    requires_grad=False,
                ),
                "scale": nn.Parameter(
                    torch.empty(out_features * in_features // 128, 1),
                    requires_grad=False,
                ),
                "zero_point": nn.Parameter(
                    torch.empty(out_features * in_features // 128, 1),
                    requires_grad=False,
                ),
            }
        )
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        self.unpacked = False

    def unpack(self):
        if self.unpacked:
            return

        self.weight = nn.Parameter(
            dequantize_tensor(
                self.weight["packed"],
                self.weight["scale"],
                self.weight["zero_point"],
                (self.out_features, self.in_features),
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
        self.unpacked = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()

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


def mlp(x: torch.Tensor, w: MLPWeights, lora: Optional[dict] = None) -> torch.Tensor:
    x0 = w.fc1(x)
    if lora is not None:
        x1 = F.linear(F.linear(x, lora["fc1"]["A"]), lora["fc1"]["B"])
        x = x0 + x1
    else:
        x = x0

    x = gelu_approx(x)

    x0 = w.fc2(x)
    if lora is not None:
        x1 = F.linear(F.linear(x, lora["fc2"]["A"]), lora["fc2"]["B"])
        x = x0 + x1
    else:
        x = x0

    return x


def moe_mlp(
    x: torch.Tensor, mlp_module: nn.Module, experts_per_token: int
) -> torch.Tensor:
    B, T, C = x.shape
    x = x.reshape(-1, C)

    # Router computation
    router_logits = mlp_module.router(x)
    topk_logits, topk_idxs = torch.topk(router_logits, experts_per_token, dim=-1)
    topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32).to(x.dtype)
    num_tokens, top_k = topk_idxs.shape

    if T == 1:
        w1_weight = mlp_module.fc1.weight
        w2_weight = mlp_module.fc2.weight

        # Flatten to process all token-expert pairs at once
        flat_idxs = topk_idxs.view(-1)  # [T*A]
        flat_weights = topk_weights.view(-1)  # [T*A]

        # Select expert weights
        w1_selected = w1_weight[flat_idxs]  # [T*A, H, D]
        w2_selected = w2_weight[flat_idxs]  # [T*A, D, H]

        # Expand input for all token-expert pairs
        x_expanded = x.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, C)  # [T*A, D]

        # First linear layer with GeGLU: [T*A, H, D] @ [T*A, D, 1] -> [T*A, H]
        x1_full = torch.bmm(w1_selected, x_expanded.unsqueeze(-1)).squeeze(
            -1
        )  # [T*A, H]
        x1, g = x1_full.chunk(2, dim=-1)
        x1 = F.gelu(x1) * (g + 1)

        # Second linear layer: [T*A, D, H] @ [T*A, H, 1] -> [T*A, D]
        expert_outs = torch.bmm(w2_selected, x1.unsqueeze(-1)).squeeze(-1)  # [T*A, D]

        # Apply weights and reshape
        weighted_outs = expert_outs * flat_weights.unsqueeze(-1)  # [T*A, D]
        weighted_outs = weighted_outs.view(num_tokens, top_k, C)  # [T, A, D]

        # Sum over experts
        mlp_out = weighted_outs.sum(dim=1)  # [T, D]
        mlp_out = mlp_out.view(B, T, C)

        return mlp_out
    else:
        out = x.new_zeros(x.size())

        for expert_id in range(mlp_module.fc1.weight.shape[0]):
            token_pos, which_k = (topk_idxs == expert_id).nonzero(as_tuple=True)
            if token_pos.numel() == 0:
                continue

            x_tok = x.index_select(0, token_pos)
            gate_tok = topk_weights[token_pos, which_k]

            h_full = F.linear(x_tok, mlp_module.fc1.weight[expert_id])
            h, g = h_full.chunk(2, dim=-1)
            h = F.gelu(h) * (g + 1)
            y = F.linear(h, mlp_module.fc2.weight[expert_id])

            y.mul_(gate_tok.unsqueeze(-1))
            out.index_add_(0, token_pos, y)

        return out.view(B, T, C)


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
