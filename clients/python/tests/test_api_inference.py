import os
import pytest
from PIL import Image
import moondream as md

TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "assets",
    "demo-1.jpg",
)


@pytest.fixture
def model():
    api_key = os.getenv("MOONDREAM_API_KEY")
    if not api_key:
        pytest.skip("MOONDREAM_API_KEY environment variable not set")
    return md.vl(api_key=api_key)


@pytest.fixture
def test_image():
    return Image.open(TEST_IMAGE_PATH)


def test_api_initialization(model):
    assert model is not None
    assert isinstance(model, md.cloud_vl.CloudVL)


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


def test_query_answering(model, test_image):
    # Test basic question answering
    result = model.query(test_image, "What is in this image?")
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_streaming_query(model, test_image):
    result = model.query(test_image, "What is in this image?", stream=True)
    assert "answer" in result

    # Test that we can iterate over the stream
    answer = ""
    for chunk in result["answer"]:
        assert isinstance(chunk, str)
        answer += chunk

    assert len(answer) > 0


@pytest.mark.skip(
    reason="API handles invalid caption lengths differently than local model"
)
def test_invalid_caption_length(model, test_image):
    with pytest.raises(ValueError, match="Model does not support caption length"):
        model.caption(test_image, length="invalid")


def test_missing_api_key():
    with pytest.raises(ValueError, match="An api_key is required for cloud inference"):
        md.vl(api_url="https://api.moondream.ai/v1")
