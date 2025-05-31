import torch
import torch.nn as nn
import math

from typing import List, Tuple, Union

from .layers import mlp

SpatialRefs = List[Union[Tuple[float, float], Tuple[float, float, float, float]]]


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
    return w.coord_encoder(fourier_features(coord, w.coord_features))


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
    return w.size_encoder(fourier_features(size, w.size_features))


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


def encode_spatial_refs(spatial_refs: SpatialRefs, w: nn.Module) -> torch.Tensor:
    """
    Takes a list of spatial references (points or regions) and encodes them into
    hidden states for input to the text model.

    Args:
        spatial_refs: List of spatial references (points or boxes)
            - Points are represented as normalized (x, y) tuples
            - Boxes are represented as normalized (x_min, y_min, x_max, y_max) tuples

    Returns:
        {"coords": torch.Tensor, "sizes": Optional[torch.Tensor]}
    """
    coords, sizes = [], []
    for ref in spatial_refs:
        if len(ref) == 2:
            coords.append(ref[0])
            coords.append(ref[1])
        else:
            x_c = (ref[0] + ref[2]) / 2
            y_c = (ref[1] + ref[3]) / 2
            width = ref[2] - ref[0]
            height = ref[3] - ref[1]
            coords.append(x_c)
            coords.append(y_c)
            sizes.append([width, height])

    coords = torch.tensor(
        coords, device=w.coord_features.device, dtype=w.coord_features.dtype
    ).view(-1, 1)
    coords = encode_coordinate(coords, w)

    if sizes:
        sizes = torch.tensor(
            sizes, device=w.size_features.device, dtype=w.size_features.dtype
        )
        sizes = encode_size(sizes, w)
    else:
        sizes = None

    return {"coords": coords, "sizes": sizes}
