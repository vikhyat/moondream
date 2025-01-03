import math
import numpy as np
import torch
import pyvips

from typing import TypedDict


def select_tiling(
    height: int, width: int, crop_size: int, max_crops: int
) -> tuple[int, int]:
    """
    Determine the optimal number of tiles to cover an image with overlapping crops.
    """
    if height <= crop_size or width <= crop_size:
        return (1, 1)

    # Minimum required tiles in each dimension
    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)

    # If minimum required tiles exceed max_crops, return proportional distribution
    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))

    # Perfect aspect-ratio tiles that satisfy max_crops
    h_tiles = math.floor(math.sqrt(max_crops * height / width))
    w_tiles = math.floor(math.sqrt(max_crops * width / height))

    # Ensure we meet minimum tile requirements
    h_tiles = max(h_tiles, min_h)
    w_tiles = max(w_tiles, min_w)

    # If we exceeded max_crops, scale down the larger dimension
    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)

    return (max(1, h_tiles), max(1, w_tiles))


class OverlapCropOutput(TypedDict):
    crops: np.ndarray
    tiling: tuple[int, int]


def overlap_crop_image(
    image: np.ndarray,
    overlap_margin: int,
    max_crops: int,
    base_size: tuple[int, int] = (378, 378),
    patch_size: int = 14,
) -> OverlapCropOutput:
    """
    Process an image using an overlap-and-resize cropping strategy with margin handling.

    This function takes an input image and creates multiple overlapping crops with
    consistent margins. It produces:
    1. A single global crop resized to base_size
    2. Multiple overlapping local crops that maintain high resolution details
    3. A patch ordering matrix that tracks correspondence between crops

    The overlap strategy ensures:
    - Smooth transitions between adjacent crops
    - No loss of information at crop boundaries
    - Proper handling of features that cross crop boundaries
    - Consistent patch indexing across the full image

    Args:
        image (np.ndarray): Input image as numpy array with shape (H,W,C)
        base_size (tuple[int,int]): Target size for crops, default (378,378)
        patch_size (int): Size of patches in pixels, default 14
        overlap_margin (int): Margin size in patch units, default 4
        max_crops (int): Maximum number of crops allowed, default 12

    Returns:
        OverlapCropOutput: Dictionary containing:
            - crops: A numpy array containing the global crop of the full image (index 0)
                followed by the overlapping cropped regions (indices 1+)
            - tiling: Tuple of (height,width) tile counts
    """
    original_h, original_w = image.shape[:2]

    # Convert margin from patch units to pixels
    margin_pixels = patch_size * overlap_margin
    total_margin_pixels = margin_pixels * 2  # Both sides

    # Calculate crop parameters
    crop_patches = base_size[0] // patch_size  # patches per crop dimension
    crop_window_patches = crop_patches - (2 * overlap_margin)  # usable patches
    crop_window_size = crop_window_patches * patch_size  # usable size in pixels

    # Determine tiling
    tiling = select_tiling(
        original_h - total_margin_pixels,
        original_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    # Pre-allocate crops.
    n_crops = tiling[0] * tiling[1] + 1  # 1 = global crop
    crops = np.zeros(
        (n_crops, base_size[0], base_size[1], image.shape[2]), dtype=np.uint8
    )

    # Resize image to fit tiling
    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    # Convert to vips for resizing
    vips_image = pyvips.Image.new_from_array(image)
    scale_x = target_size[1] / image.shape[1]
    scale_y = target_size[0] / image.shape[0]
    resized = vips_image.resize(scale_x, vscale=scale_y)
    image = resized.numpy()

    # Create global crop
    scale_x = base_size[1] / vips_image.width
    scale_y = base_size[0] / vips_image.height
    global_vips = vips_image.resize(scale_x, vscale=scale_y)
    crops[0] = global_vips.numpy()

    for i in range(tiling[0]):
        for j in range(tiling[1]):
            # Calculate crop coordinates
            y0 = i * crop_window_size
            x0 = j * crop_window_size

            # Extract crop with padding if needed
            y_end = min(y0 + base_size[0], image.shape[0])
            x_end = min(x0 + base_size[1], image.shape[1])

            crop_region = image[y0:y_end, x0:x_end]
            crops[
                1 + i * tiling[1] + j, : crop_region.shape[0], : crop_region.shape[1]
            ] = crop_region

    return {"crops": crops, "tiling": tiling}


def reconstruct_from_crops(
    crops: torch.Tensor,
    tiling: tuple[int, int],
    overlap_margin: int,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Reconstruct the original image from overlapping crops into a single seamless image.

    Takes a list of overlapping image crops along with their positional metadata and
    reconstructs them into a single coherent image by carefully stitching together
    non-overlapping regions. Handles both numpy arrays and PyTorch tensors.

    Args:
        crops: List of image crops as numpy arrays or PyTorch tensors with shape
            (H,W,C)
        tiling: Tuple of (height,width) indicating crop grid layout
        patch_size: Size in pixels of each patch, default 14
        overlap_margin: Number of overlapping patches on each edge, default 4

    Returns:
        Reconstructed image as numpy array or PyTorch tensor matching input type,
        with shape (H,W,C) where H,W are the original image dimensions
    """
    tiling_h, tiling_w = tiling
    crop_height, crop_width = crops[0].shape[:2]
    margin_pixels = overlap_margin * patch_size

    # Calculate output size (only adding margins once)
    output_h = (crop_height - 2 * margin_pixels) * tiling_h + 2 * margin_pixels
    output_w = (crop_width - 2 * margin_pixels) * tiling_w + 2 * margin_pixels

    reconstructed = torch.zeros(
        (output_h, output_w, crops[0].shape[2]),
        device=crops[0].device,
        dtype=crops[0].dtype,
    )

    for i, crop in enumerate(crops):
        tile_y = i // tiling_w
        tile_x = i % tiling_w

        # For each tile, determine which part to keep
        # Keep left margin only for first column
        x_start = 0 if tile_x == 0 else margin_pixels
        # Keep right margin only for last column
        x_end = crop_width if tile_x == tiling_w - 1 else crop_width - margin_pixels
        # Keep top margin only for first row
        y_start = 0 if tile_y == 0 else margin_pixels
        # Keep bottom margin only for last row
        y_end = crop_height if tile_y == tiling_h - 1 else crop_height - margin_pixels

        # Calculate where this piece belongs in the output
        out_x = tile_x * (crop_width - 2 * margin_pixels)
        out_y = tile_y * (crop_height - 2 * margin_pixels)

        # Place the piece
        reconstructed[
            out_y + y_start : out_y + y_end, out_x + x_start : out_x + x_end
        ] = crop[y_start:y_end, x_start:x_end]

    return reconstructed
