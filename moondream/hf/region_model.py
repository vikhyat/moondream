import torch
import torch.nn as nn

from .fourier_features import FourierFeatures


class MLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

        torch.nn.init.kaiming_normal_(
            self.fc1.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_normal_(
            self.fc2.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class RegionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.coordinate_features = FourierFeatures(1, 256)
        self.coordinate_encoder = nn.Linear(256, 2048)
        self.size_features = FourierFeatures(2, 512)
        self.size_encoder = nn.Linear(512, 2048)

        self.coordinate_decoder = MLP(2048, 8192, 1024)
        self.size_decoder = MLP(2048, 8192, 2048)

    def encode_coordinate(self, coordinate):
        return self.coordinate_encoder(self.coordinate_features(coordinate))

    def encode_size(self, size):
        return self.size_encoder(self.size_features(size))

    def decode_coordinate(self, logit):
        return self.coordinate_decoder(logit)

    def decode_size(self, logit):
        o = self.size_decoder(logit)
        return o.view(-1, 2, 1024)

    def encode(self, position, size):
        c = self.encode_coordinate(position.view(2, 1)).view(2, 2048)
        return torch.stack([c[0], c[1], self.encode_size(size)], dim=0)

    def decode(self, position_logits, size_logits):
        return (
            self.decode_coordinate(position_logits),
            self.decode_size(size_logits),
        )
