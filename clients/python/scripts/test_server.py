import requests
from PIL import Image
import io
import json


def read_stream_response(response):
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if "chunk" in data:
                    print(data["chunk"], end="", flush=True)
                if data.get("completed"):
                    print()
                    break
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")


# run `moondream serve` first
def test_server(base_url="http://localhost:3281"):
    # Test health endpoint
    print("\nTesting /health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    # Load a test image
    try:
        test_image = Image.open("../../assets/demo-1.jpg")
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()
    except Exception as e:
        print(f"Error loading test image: {e}")
        return

    # Test caption endpoint (streaming)
    print("\nTesting /caption streaming...")
    response = requests.post(
        f"{base_url}/caption?stream=true",
        data=img_byte_arr,
        headers={"Content-Type": "image/jpeg"},
        stream=True,
    )
    print(f"Status: {response.status_code}")
    read_stream_response(response)

    # Test caption endpoint (normal)
    print("\nTesting /caption endpoint...")
    response = requests.post(
        f"{base_url}/caption", data=img_byte_arr, headers={"Content-Type": "image/jpeg"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    # Test query endpoint (normal)
    print("\nTesting /query endpoint...")
    response = requests.post(
        f"{base_url}/query?question=What is in this image?",
        data=img_byte_arr,
        headers={"Content-Type": "image/jpeg"},
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    # Test query endpoint (streaming)
    print("\nTesting /query streaming...")
    response = requests.post(
        f"{base_url}/query?question=What is in this image?&stream=true",
        data=img_byte_arr,
        headers={"Content-Type": "image/jpeg"},
        stream=True,
    )
    print(f"Status: {response.status_code}")
    read_stream_response(response)

    # Test detect endpoint
    print("\nTesting /detect endpoint...")
    response = requests.post(
        f"{base_url}/detect?object=person",
        data=img_byte_arr,
        headers={"Content-Type": "image/jpeg"},
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    # Test point endpoint
    print("\nTesting /point endpoint...")
    response = requests.post(
        f"{base_url}/point?object=person",
        data=img_byte_arr,
        headers={"Content-Type": "image/jpeg"},
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")


if __name__ == "__main__":
    test_server()
