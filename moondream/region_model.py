import torch
import torch.nn as nn
from .fourier_features import FourierFeatures

class RegionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.position_features = FourierFeatures(2, 256)
        self.position_encoder = nn.Linear(256, 2048)
        self.size_features = FourierFeatures(2, 256)
        self.size_encoder = nn.Linear(256, 2048)

        self.position_decoder = nn.Linear(2048, 2)
        self.size_decoder = nn.Linear(2048, 2)
        self.confidence_decoder = nn.Linear(2048, 1)

    def encode_position(self, position):
        return self.position_encoder(self.position_features(position))

    def encode_size(self, size):
        return self.size_encoder(self.size_features(size))

    def decode_position(self, x):
        return self.position_decoder(x)

    def decode_size(self, x):
        return self.size_decoder(x)

    def decode_confidence(self, x):
        return self.confidence_decoder(x)

    def encode(self, position, size):
        return torch.stack(
            [self.encode_position(position), self.encode_size(size)], dim=0
        )

    def decode(self, position_logits, size_logits):
        return (
            self.decode_position(position_logits),
            self.decode_size(size_logits),
            self.decode_confidence(size_logits),
        )
