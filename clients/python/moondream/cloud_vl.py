import json
import os
import urllib.request
import base64
from PIL import Image
from io import BytesIO
from typing import Union, Optional, Literal


from .types import (
    VLM,
    CaptionOutput,
    EncodedImage,
    QueryOutput,
    DetectOutput,
    SamplingSettings,
)


class CloudVL(VLM):
    @classmethod
    def from_api_key(cls, api_key: str) -> "CloudVL":
        """Initialize a CloudVL instance with an API key.

        Args:
            api_key (str): The API key for authentication
        Returns:
            CloudVL: An initialized CloudVL instance
        """
        return cls(api_key=api_key)

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.api_url = "https://api.moondream.ai/v1"

    def _encode_image(self, image: Image.Image) -> str:
        try:
            # Resize image preserving aspect ratio
            width, height = image.size
            if width > height:
                if width > 768:
                    new_width = 768
                    new_height = int(height * (768 / width))
                    image = image.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )
            else:
                if height > 768:
                    new_height = 768
                    new_width = int(width * (768 / height))
                    image = image.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )

            if image.mode != "RGB":
                image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            raise ValueError(f"Failed to convert image to JPEG: {str(e)}") from e

    def encode_image(self, image: Union[Image.Image, EncodedImage]) -> EncodedImage:
        raise NotImplementedError("encode_image is not supported for cloud inference")

    def _stream_response(self, req):
        """Helper function to stream response chunks from the API."""
        with urllib.request.urlopen(req) as response:
            for line in response:
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        if "chunk" in chunk:
                            yield chunk["chunk"]
                        if chunk.get("completed"):
                            break
                    except json.JSONDecodeError:
                        continue

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: Literal["short", "normal"] = "normal",
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> CaptionOutput:
        if isinstance(image, EncodedImage):
            raise ValueError("EncodedImage not supported for cloud inference")

        payload = {
            "image_url": self._encode_image(image),
            "length": length,
            "stream": stream,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {"X-Moondream-Auth": self.api_key, "Content-Type": "application/json"}
        req = urllib.request.Request(
            f"{self.api_url}/caption",
            data=data,
            headers=headers,
        )

        def generator():
            for chunk in self._stream_response(req):
                yield chunk

        if stream:
            return {"caption": generator()}

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {"caption": result.get("caption", "")}

    def query(
        self,
        image: Union[Image.Image, EncodedImage],
        question: str,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> QueryOutput:
        if isinstance(image, EncodedImage):
            raise ValueError("EncodedImage not supported for cloud inference")

        payload = {
            "image_url": self._encode_image(image),
            "question": question,
            "stream": stream,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {"X-Moondream-Auth": self.api_key, "Content-Type": "application/json"}
        req = urllib.request.Request(
            f"{self.api_url}/query",
            data=data,
            headers=headers,
        )

        if stream:
            return {"answer": self._stream_response(req)}

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {"answer": result.get("answer", "")}

    def detect(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
    ) -> DetectOutput:
        if isinstance(image, EncodedImage):
            raise ValueError("EncodedImage not supported for cloud inference")

        payload = {"image_url": self._encode_image(image), "object": object}

        data = json.dumps(payload).encode("utf-8")
        headers = {"X-Moondream-Auth": self.api_key, "Content-Type": "application/json"}
        req = urllib.request.Request(
            f"{self.api_url}/detect",
            data=data,
            headers=headers,
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["objects"]
