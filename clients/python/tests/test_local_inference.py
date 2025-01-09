import os
import pytest
from PIL import Image
import moondream as md

MODEL_URL = "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "test_data", "moondream-0_5b-int8.mf"
)
TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "assets",
    "demo-1.jpg",
)


@pytest.fixture(scope="session", autouse=True)
def download_model():
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        import requests
        import gzip
        import io

        # Download the model file
        print("Downloading model file...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        # Read the gzipped content into memory
        content = response.content

        # Decompress and save
        print("Decompressing model file...")
        with gzip.open(io.BytesIO(content), "rb") as f_in:
            with open(MODEL_PATH, "wb") as f_out:
                f_out.write(f_in.read())
        print("Model file ready.")


@pytest.fixture
def model():
    return md.vl(model=MODEL_PATH)


@pytest.fixture
def test_image():
    return Image.open(TEST_IMAGE_PATH)


def test_model_initialization(model):
    assert model is not None
    assert hasattr(model, "vision_encoder")
    assert hasattr(model, "text_decoder")
    assert hasattr(model, "tokenizer")


def test_image_encoding(model, test_image):
    encoded_image = model.encode_image(test_image)
    assert encoded_image is not None
    assert hasattr(encoded_image, "pos")
    assert hasattr(encoded_image, "kv_cache")


def test_image_captioning(model, test_image):
    # Test normal length caption
    result = model.caption(test_image, length="normal")
    assert "caption" in result
    assert isinstance(result["caption"], str)
    assert len(result["caption"]) > 0

    # Test short length caption
    result = model.caption(test_image, length="short")
    assert "caption" in result
    assert isinstance(result["caption"], str)
    assert len(result["caption"]) > 0


def test_streaming_caption(model, test_image):
    result = model.caption(test_image, stream=True)
    assert "caption" in result

    # Test that we can iterate over the stream
    caption = ""
    for chunk in result["caption"]:
        assert isinstance(chunk, str)
        caption += chunk

    assert len(caption) > 0


def test_reuse_encoded_image(model, test_image):
    # Test that we can reuse an encoded image for multiple operations
    encoded_image = model.encode_image(test_image)

    # Generate two captions using the same encoded image
    result1 = model.caption(encoded_image)
    result2 = model.caption(encoded_image)

    assert result1["caption"] == result2["caption"]


def test_invalid_caption_length(model, test_image):
    with pytest.raises(ValueError, match="Model does not support caption length"):
        model.caption(test_image, length="invalid")


def test_invalid_model_path():
    with pytest.raises(
        ValueError,
        match="Unsupported model filetype. Please use a .safetensors for GPU use or .mf for CPU use.",
    ):
        md.vl(model="invalid/path/to/model.bin")
