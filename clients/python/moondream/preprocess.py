from typing import List, Tuple

import numpy as np
from PIL import Image


def im_resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: int = Image.Resampling.BICUBIC,
) -> Image.Image:
    return image.resize(size, resample=resample)


def normalize(
    image: np.ndarray,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
) -> np.ndarray:
    """
    Normalize an image array.
    """
    return (image - np.array(mean)) / np.array(std)


def create_patches(image: Image.Image, image_patch_size=378) -> np.ndarray:
    """
    Split the given image into a variable number of patches depending upon its
    resolution.
    """
    # Start off with the global patch.
    patches = [im_resize(image, (image_patch_size, image_patch_size))]

    # Find the closest resolution template.
    res_templates = [(1, 2), (2, 1), (2, 2)]
    im_width, im_height = image.size
    max_dim = max(im_width, im_height)
    if max_dim < image_patch_size * 1.4:
        # If the image is already small, we just do a single patch that is a
        # duplicate of the global patch. This creates a small amount of
        # redundant computation now, but it is simpler and future-proofs us
        # if/when we condition the vision encoder on the patch type.
        patches.append(patches[0])
    else:
        aspect_ratio = im_width / im_height
        res_template = min(
            res_templates, key=lambda size: abs((size[1] / size[0]) - aspect_ratio)
        )
        # TODO: Actually implement patching... just going to put in the global
        # patch for now to make progress on other aspects.
        patches.append(patches[0])

    return np.stack(
        [
            normalize(
                (np.array(patch_img) / 255.0),
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ).transpose(2, 0, 1)
            for patch_img in patches
        ],
        dtype=np.float16,
    )
