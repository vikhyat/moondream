import moondream as md
from PIL import Image


def test_torch_model():
    # Initialize with local model
    vl = md.vl(
        model="/Users/caleb/Projects/moondream/moondream-playground-inf/src/ai-models/05/moondream-01-08-2025.safetensors"
    )

    # Load an image
    image = Image.open("/Users/caleb/Projects/moondream/moondream/assets/demo-1.jpg")

    # Test caption
    print("\nTesting caption:")
    result = vl.caption(image, length="normal")
    print(f"Caption: {result['caption']}")

    # Test query
    print("\nTesting query:")
    result = vl.query(image, "What is in this image?")
    print(f"Answer: {result['answer']}")

    # Test detect
    print("\nTesting detect:")
    result = vl.detect(image, "person")
    print(f"Found {len(result['objects'])} instances")

    # Test point
    print("\nTesting point:")
    result = vl.point(image, "face")
    print(f"Found {len(result['points'])} points")


if __name__ == "__main__":
    test_torch_model()
