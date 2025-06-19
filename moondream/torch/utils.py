import numpy as np
from pathlib import Path
import os

def remove_outlier_points(points_tuples, k_nearest=2, threshold=2.0):
    """
    Robust outlier detection for list of (x,y) tuples.
    Only requires numpy.

    Args:
        points_tuples: list of (x,y) tuples
        k_nearest: number of neighbors to consider
        threshold: multiplier for median distance

    Returns:
        list: filtered list of (x,y) tuples with outliers removed
        list: list of booleans indicating which points were kept (True = kept)
    """
    points = np.array(points_tuples)
    n_points = len(points)

    # Calculate pairwise distances manually
    dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Euclidean distance between points i and j
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Get k nearest neighbors' distances
    k = min(k_nearest, n_points - 1)
    neighbor_distances = np.partition(dist_matrix, k, axis=1)[:, :k]
    avg_neighbor_dist = np.mean(neighbor_distances, axis=1)

    # Calculate mask using median distance
    median_dist = np.median(avg_neighbor_dist)
    mask = avg_neighbor_dist <= threshold * median_dist

    # Return filtered tuples and mask
    filtered_tuples = [t for t, m in zip(points_tuples, mask) if m]
    return filtered_tuples


def hf_hub_dir() -> Path:
    """Return the HuggingFace hub cache directory."""

    directory = os.getenv("HF_HUB_CACHE")
    if not directory:
        directory = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return directory


def rename_state_dict(state_dict: dict) -> dict:
    """Rename raw weights name for HF."""
    
    rename_rules = [
        ("text_model.transformer.h", "text.blocks"),
        (".mixer", ".attn"),
        (".out_proj", ".proj"),
        (".Wqkv", ".qkv"),
    ]

    new_state_dict = {}
    for key, tensor in state_dict.items():
        new_key = key
        for old, new in rename_rules:
            if old in new_key:
                new_key = new_key.replace(old, new)
        new_state_dict[new_key] = tensor

    return new_state_dict