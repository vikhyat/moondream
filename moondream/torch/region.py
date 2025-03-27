import torch
import torch.nn as nn
import math

from .layers import linear, mlp


def fourier_features(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Applies Fourier feature mapping to input tensor x using frequency matrix w. This
    projects inputs through sinusoidal functions to create higher dimensional features
    that help mitigate spectral bias - the tendency of neural networks to learn
    low-frequency functions more easily than high-frequency ones. By explicitly
    mapping inputs to higher frequencies through sin/cos transformations, we enable
    better learning of fine details and higher frequency patterns.

    Args:
        x: Input tensor to transform
        w: Matrix of frequencies for the Fourier features transformation

    Returns:
        Concatenated cosine and sine transformed features as a tensor
    """
    f = 2 * math.pi * x @ w
    return torch.cat([f.cos(), f.sin()], dim=-1)


def encode_coordinate(coord: torch.Tensor, w: nn.Module) -> torch.Tensor:
    """
    Takes as input a tensor containing a single float coordinate value (x or y)
    and encodes it into hidden states for input to the text model.

    Args:
        coord: Tensor with single float coordinate value

    Returns:
        Encoded hidden states tensor for input to text model
    """
    return linear(fourier_features(coord, w.coord_features), w.coord_encoder)


def decode_coordinate(hidden_state: torch.Tensor, w: nn.Module) -> torch.Tensor:
    """
    Takes as input the last hidden state from the text model and outputs a single logit
    representing either an x or y coordinate prediction.

    Args:
        hidden_state: The final hidden state tensor from the text model.

    Returns:
        A single logit representing the predicted coordinate value (x or y)
    """
    return mlp(hidden_state, w.coord_decoder)


def encode_size(size: torch.Tensor, w: nn.Module) -> torch.Tensor:
    """
    Takes a tensor containing width and height values and encodes them into
    hidden states for input to the text model.

    Args:
        size: Tensor with two floats for width and height

    Returns:
        Encoded hidden states tensor for input to text model
    """
    return linear(fourier_features(size, w.size_features), w.size_encoder)


def decode_size(hidden_state: torch.Tensor, w: nn.Module) -> torch.Tensor:
    """
    Takes as input the last hidden state from the text model and outputs logits
    for 1024 bins representing width and height in log-scale.

    The bins are distributed according to the formula:
    bin = (log2(size) + 10.0) / 10.0 * 1023.0
    where size values are clamped to be at least 1/1024.

    To convert from bin back to size:
    size = 2^((bin / 1023.0) * 10.0 - 10.0)

    Args:
        hidden_state: The final hidden state tensor from the text model.

    Returns:
        A tensor containing logits for 1024 bins for width and height.
        Shape is (2, 1024) where the first dimension corresponds to width and height.
    """
    return mlp(hidden_state, w.size_decoder).view(2, -1)
