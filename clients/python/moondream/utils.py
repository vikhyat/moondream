from pathlib import Path
from PIL import Image
import base64
import os
import traceback
from io import BytesIO
from typing import Union

BASE_URL = "http://localhost:4002"
API_VERSION = "v1"


def validate_image(image: Union[Image.Image, Path, str]) -> BytesIO:
    """
    Validates and processes an image input, converting it to a JPEG format BytesIO object.

    This function accepts various image input formats and performs the following:
    1. Validates the input image
    2. Converts RGBA images to RGB format
    3. Converts the image to a JPEG format BytesIO object

    Args:
        image (Union[Image.Image, Path, str]): The input image in one of these formats:
            - PIL Image object
            - Path object pointing to an image file
            - String containing either:
                - A file path to an image
                - A base64 encoded image string

    Returns:
        BytesIO: A BytesIO object containing the JPEG-encoded image data

    Raises:
        ValueError: If the image is invalid, cannot be found, or cannot be processed.
            Specific error messages include:
            - Invalid file path
            - Invalid base64 string
            - Unsupported image format
            - File not found
            - Invalid input type
    """
    pil_image = None
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        # Check if the string is a file path
        if os.path.exists(image):
            pil_image = Image.open(image)
        else:
            # Handle base64 encoded image string
            try:
                image_data = base64.b64decode(image)
                pil_image = Image.open(BytesIO(image_data))
            except Exception:
                raise ValueError(
                    f"{image[:75]}... is not a valid image file or base64 encoded image string. \nError: {traceback.format_exc()}"
                )
    elif isinstance(image, Path):
        # Handle Path object
        if not image.exists():
            raise ValueError(f"Image file not found: {image}")
        try:
            pil_image = Image.open(image)
        except Exception:
            raise ValueError(
                f"Failed to open image file at {image}.\nError: {traceback.format_exc()}"
            )

    if pil_image is None:
        raise ValueError(
            "Please provide a valid PIL Image object, file path, or base64 encoded image string."
        )

    # Convert RGBA to RGB if necessary
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    # Convert PIL Image to BytesIO object
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr
