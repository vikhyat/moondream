# Adopted from https://github.com/crowsonkb/k-diffusion/blob/transformer-model-v2/k_diffusion/layers.py

import torch
import torch.nn as nn
import math


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer(
            "weight", torch.randn([out_features // 2, in_features]) * std
        )

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
