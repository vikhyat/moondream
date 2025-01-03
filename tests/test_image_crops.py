import numpy as np
import torch
from moondream.torch.image_crops import overlap_crop_image, reconstruct_from_crops


def test_overlap_crop_basic():
    # Create a test image
    test_image = np.zeros((800, 600, 3), dtype=np.uint8)
    # Add a recognizable pattern - white rectangle in the middle
    test_image[300:500, 200:400] = 255

    result = overlap_crop_image(test_image, overlap_margin=4, max_crops=12)

    # Check basic properties
    assert result["crops"][0].shape == (378, 378, 3)
    assert len(result["crops"]) > 1
    assert all(crop.shape == (378, 378, 3) for crop in result["crops"])
    assert len(result["tiling"]) == 2


def test_overlap_crop_small_image():
    # Test with image smaller than crop size
    test_image = np.zeros((300, 200, 3), dtype=np.uint8)
    result = overlap_crop_image(test_image, overlap_margin=4, max_crops=12)

    # Should still produce valid output
    assert result["crops"][0].shape == (378, 378, 3)
    assert len(result["crops"]) == 2
    assert result["tiling"] == (1, 1)


def test_reconstruction():
    # Create a test image
    test_image = np.zeros((800, 600, 3), dtype=np.uint8)
    # Add a recognizable pattern
    test_image[300:500, 200:400] = 255

    # Crop and reconstruct
    result = overlap_crop_image(test_image, overlap_margin=4, max_crops=12)
    crops_tensor = [torch.from_numpy(crop) for crop in result["crops"][1:]]
    reconstructed = reconstruct_from_crops(
        crops_tensor, result["tiling"], overlap_margin=4
    )

    # Convert back to numpy for comparison
    reconstructed_np = reconstructed.numpy()

    # The reconstructed image should be similar to the input
    # We can't expect exact equality due to resizing operations
    # but the white rectangle should still be visible in the middle
    center_reconstructed = reconstructed_np[
        reconstructed_np.shape[0] // 2 - 100 : reconstructed_np.shape[0] // 2 + 100,
        reconstructed_np.shape[1] // 2 - 100 : reconstructed_np.shape[1] // 2 + 100,
    ].mean()

    # The center region should be significantly brighter than the edges
    assert center_reconstructed > reconstructed_np[:100, :100].mean() + 100
