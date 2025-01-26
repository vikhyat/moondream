import numpy as np
from moondream.torch import image_crops
from moondream.torch.image_crops import overlap_crop_image

def create_test_pattern(height, width):
    """Create a test image with a recognizable pattern."""
    # Create gradient patterns
    y = np.linspace(0, 1, height)[:, np.newaxis]
    x = np.linspace(0, 1, width)[np.newaxis, :]
    
    # Broadcast to create 2D patterns
    y_grid = np.broadcast_to(y, (height, width))
    x_grid = np.broadcast_to(x, (height, width))
    
    # Create RGB channels with different patterns
    r = (y_grid * 255).astype(np.uint8)  # Vertical gradient
    g = (x_grid * 255).astype(np.uint8)  # Horizontal gradient
    b = ((x_grid + y_grid) * 127).astype(np.uint8)  # Diagonal gradient
    
    return np.dstack([r, g, b])

def compare_results(pyvips_result, pil_result):
    """Compare results from PyVips and PIL implementations."""
    pyvips_crops = pyvips_result["crops"]
    pil_crops = pil_result["crops"]
    
    # Compare number of crops and tiling
    assert pyvips_result["tiling"] == pil_result["tiling"], \
        f"Tiling mismatch: PyVips {pyvips_result['tiling']} vs PIL {pil_result['tiling']}"
    assert len(pyvips_crops) == len(pil_crops), \
        f"Number of crops mismatch: PyVips {len(pyvips_crops)} vs PIL {len(pil_crops)}"
    
    # Compare each crop
    max_diff = 0
    mean_diff = 0
    
    for i, (vips_crop, pil_crop) in enumerate(zip(pyvips_crops, pil_crops)):
        # Convert to float for comparison
        vips_float = vips_crop.astype(float)
        pil_float = pil_crop.astype(float)
        
        # Calculate differences
        diff = np.abs(vips_float - pil_float)
        crop_max_diff = np.max(diff)
        crop_mean_diff = np.mean(diff)
        
        max_diff = max(max_diff, crop_max_diff)
        mean_diff += crop_mean_diff / len(pyvips_crops)
        
        print(f"Crop {i}:")
        print(f"  Max pixel difference: {crop_max_diff:.2f}")
        print(f"  Mean pixel difference: {crop_mean_diff:.2f}")
    
    print(f"\nOverall:")
    print(f"  Maximum pixel difference: {max_diff:.2f}")
    print(f"  Average pixel difference: {mean_diff:.2f}")
    
    # Fail if differences are too large
    assert max_diff < 5.0, f"Max pixel difference ({max_diff:.2f}) is too large"
    assert mean_diff < 2.0, f"Mean pixel difference ({mean_diff:.2f}) is too large"

def test_image_crops():
    # Create a test image with a known pattern
    test_image = create_test_pattern(300, 400)
    
    # Get results using PyVips
    image_crops.HAVE_PYVIPS = True
    pyvips_result = overlap_crop_image(
        image=test_image,
        overlap_margin=4,
        max_crops=12,
        base_size=(378, 378),
        patch_size=14
    )
    print("\nPyVips test completed")
    
    # Get results using PIL
    image_crops.HAVE_PYVIPS = False
    pil_result = overlap_crop_image(
        image=test_image,
        overlap_margin=4,
        max_crops=12,
        base_size=(378, 378),
        patch_size=14
    )
    print("\nPIL test completed")
    
    # Compare results
    print("\nComparing results:")
    compare_results(pyvips_result, pil_result)

if __name__ == "__main__":
    test_image_crops()