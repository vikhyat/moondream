import os
from PIL import Image
import moondream as md


def test_cloud_client():
    # Initialize client
    api_key = os.getenv("MOONDREAM_API_KEY")
    if not api_key:
        raise ValueError("MOONDREAM_API_KEY environment variable not set")

    client = md.VL(api_key=api_key)

    # Load a test image
    image_path = "/Users/caleb/Projects/moondream/moondream/data/raven_01.webp"
    image = Image.open(image_path)

    # Test caption
    print("\nTesting non-streaming caption...")
    caption = client.caption(image)
    print(f"Caption result: {caption['caption']}")

    # Test streaming caption
    print("\nTesting streaming caption...")
    for tok in client.caption(image, stream=True)["caption"]:
        print(tok, end="", flush=True)
    print()

    # Test long caption
    print("\nTesting short caption...")
    caption = client.caption(image, length="short")
    print(f"Caption result: {caption['caption']}")

    # Test query
    print("\nTesting non-streaming query...")
    question = "What type of animal is this? Be detailed."
    answer = client.query(image, question)
    print(f"Query result: {answer['answer']}")

    # Test streaming query
    print("\nTesting streaming query...")
    for chunk in client.query(image, question, stream=True)["answer"]:
        print(chunk, end="", flush=True)
    print()

    # Test detect
    print("\nTesting detect...")
    object_to_detect = "bird"
    objects = client.detect(image, object_to_detect)
    print(f"Detect result: {objects}")


if __name__ == "__main__":
    test_cloud_client()
