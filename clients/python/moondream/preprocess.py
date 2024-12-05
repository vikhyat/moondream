from typing import List, Tuple

import numpy as np
from PIL import Image


def im_resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: int = Image.Resampling.BICUBIC,
) -> Image.Image:
    return image.resize(size, resample=resample)


def adaptive_avg_pool2d(x, output_size):
    """Applies 2D adaptive average pooling over an input signal.

    Resizes input to a target size by averaging values in local neighborhoods.
    The neighborhoods are computed to evenly cover the input image while
    maintaining approximately equal size. Similar to PyTorch's
    adaptive_avg_pool2d but expects input shape (H,W,C) rather than (N,C,H,W).

    Args:
        x: Input tensor of shape (height, width, channels)
        output_size: Target output size. Can be:
            - Single integer for square output (size, size)
            - Tuple of two ints (out_height, out_width)

    Returns:
        Tensor of shape (out_height, out_width, channels)

    Example:
        >>> img = np.random.randn(32, 32, 3)  # 32x32 RGB image
        >>> pooled = adaptive_avg_pool2d(img, (7, 7))  # Resize to 7x7
        >>> pooled.shape
        (7, 7, 3)
    """
    height, width, channels = x.shape

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    out_height, out_width = output_size

    stride_h = height // out_height
    stride_w = width // out_width
    kernel_h = height - (out_height - 1) * stride_h
    kernel_w = width - (out_width - 1) * stride_w

    output = np.zeros((out_height, out_width, channels), dtype=x.dtype)

    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride_h
            h_end = h_start + kernel_h
            w_start = j * stride_w
            w_end = w_start + kernel_w
            output[i, j, :] = x[h_start:h_end, w_start:w_end, :].mean(axis=(0, 1))

    return output


def normalize(
    image: np.ndarray,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
) -> np.ndarray:
    """
    Normalize an image array.
    """
    return (image - np.array(mean)) / np.array(std)


def create_patches(
    image: Image.Image, image_patch_size=378
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Split the given image into a variable number of patches depending upon its
    resolution. Returns the patches as a numpy array, and the selected patching
    template as a tuple of (rows, cols).
    """
    image = image.convert("RGB")

    # Start off with the global patch.
    patches = [im_resize(image, (image_patch_size, image_patch_size))]

    # Find the closest resolution template.
    #
    # (1, 2)              (2, 1)              (2, 2)
    # +-------+-------+   +-----------+       +-------+-------+
    # |   1   |   2   |   |     1     |       |   1   |   2   |
    # +-------+-------+   +-----------+       +-------+-------+
    #                     |     2     |       |   3   |   4   |
    #                     +-----------+       +-------+-------+
    res_templates = [(1, 2), (2, 1), (2, 2)]

    im_width, im_height = image.size
    max_dim = max(im_width, im_height)
    if max_dim < image_patch_size * 1.4:
        # If the image is already small, we avoid adding an extra patch
        # here to avoid redundant computation in the vision encoder, and
        # instead copy the global patch features after running the vision
        # encoder, before passing it through the vision projection.
        selected_template = (1, 1)
    else:
        aspect_ratio = im_width / im_height
        selected_template = min(
            res_templates, key=lambda size: abs((size[1] / size[0]) - aspect_ratio)
        )

        patch_width = im_width // selected_template[1]
        patch_height = im_height // selected_template[0]

        for row in range(selected_template[0]):
            for col in range(selected_template[1]):
                x_min = col * patch_width
                y_min = row * patch_height
                x_max = x_min + patch_width
                y_max = y_min + patch_height
                patches.append(
                    im_resize(
                        image.crop((x_min, y_min, x_max, y_max)),
                        (image_patch_size, image_patch_size),
                    )
                )

    return (
        np.stack(
            [
                normalize(
                    (np.array(patch_img) / 255.0),
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ).transpose(2, 0, 1)
                for patch_img in patches
            ],
            dtype=np.float16,
        ),
        selected_template,
    )
