import os
from PIL import Image
import moondream as md


def test_cloud_client():
    # Initialize client
    api_key = os.environ["MOONDREAM_API_KEY"]
    client = md.VL(api_key=api_key)

    # Load a test image
    image_path = "../../assets/demo-1.jpg"
    image = Image.open(image_path)
    encoded_image = client.encode_image(image)

    # Test caption
    print("\nTesting non-streaming caption...")
    caption = client.caption(image)
    print(f"Caption result: {caption['caption']}")

    # Test streaming caption
    print("\nTesting streaming caption...")
    for tok in client.caption(encoded_image, stream=True)["caption"]:
        print(tok, end="", flush=True)
    print()

    # Test long caption
    print("\nTesting short caption...")
    caption = client.caption(image, length="short")
    print(f"Caption result: {caption['caption']}")

    # Test query
    print("\nTesting non-streaming query...")
    question = "What is the character eating?"
    answer = client.query(encoded_image, question)
    print(f"Query result: {answer['answer']}")

    # Test streaming query
    print("\nTesting streaming query...")
    for chunk in client.query(image, question, stream=True)["answer"]:
        print(chunk, end="", flush=True)
    print()

    # Test detect
    print("\nTesting detect...")
    object_to_detect = "burger"
    objects = client.detect(encoded_image, object_to_detect)
    print(f"Detect result: {objects}")


if __name__ == "__main__":
    test_cloud_client()
