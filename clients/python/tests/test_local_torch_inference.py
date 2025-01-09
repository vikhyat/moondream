import os
import pytest
from PIL import Image
import moondream as md

MODEL_PATH = "/Users/caleb/Projects/moondream/moondream-playground-inf/src/ai-models/05/moondream-01-08-2025.safetensors"
TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "assets",
    "demo-1.jpg",
)


@pytest.fixture
def model():
    return md.vl(model=MODEL_PATH)


@pytest.fixture
def test_image():
    return Image.open(TEST_IMAGE_PATH)


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

    # Test streaming caption
    result = model.caption(test_image, stream=True)
    assert "caption" in result

    # Test that we can iterate over the stream
    num_chunks = 0
    caption = ""
    for chunk in result["caption"]:
        assert isinstance(chunk, str)
        caption += chunk
        num_chunks += 1

    assert len(caption) > 0
    assert num_chunks > 1


def test_query(model, test_image):
    result = model.query(test_image, "What is in this image?")
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0

    # Test streaming query
    result = model.query(test_image, "What is in this image?", stream=True)
    assert "answer" in result

    # Test that we can iterate over the stream
    num_chunks = 0
    answer = ""
    for chunk in result["answer"]:
        assert isinstance(chunk, str)
        answer += chunk
        num_chunks += 1

    assert num_chunks > 1
    assert len(answer) > 0


def test_detect(model, test_image):
    result = model.detect(test_image, "person")
    assert "objects" in result
    assert isinstance(result["objects"], list)


def test_point(model, test_image):
    result = model.point(test_image, "face")
    assert "points" in result
    assert isinstance(result["points"], list)
