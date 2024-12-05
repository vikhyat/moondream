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
    def from_api_key(
        cls, api_key: str, api_url: str = None, version: str = "v1"
    ) -> "CloudVL":
        """Initialize a CloudVL instance with an API key.

        Args:
            api_key (str): The API key for authentication
            api_url (str, optional): Custom API URL. Defaults to None.
            version (str, optional): API version. Defaults to "v1".

        Returns:
            CloudVL: An initialized CloudVL instance
        """
        return cls(api_key=api_key, api_url=api_url, version=version)

    def __init__(self, api_key: str = None, api_url: str = None, version: str = "v1"):
        self.api_key = api_key or os.getenv("MOONDREAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set MOONDREAM_API_KEY env var or pass api_key"
            )

        self.api_url = api_url or f"https://api.moondream.ai/{version}"
        self.client = urllib.request.urlopen
        self.headers = {"X-Moondream-Auth": self.api_key}

    def _encode_image(self, image: Image.Image) -> str:
        try:
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            raise ValueError(f"Failed to convert image to PNG: {str(e)}")

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
                        if chunk.get("completed"):
                            break
                        yield chunk.get("chunk", "")
                    except json.JSONDecodeError:
                        continue

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: Literal["long", "short"] = "long",
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
        req = urllib.request.Request(
            f"{self.api_url}/caption",
            data=data,
            headers={**self.headers, "Content-Type": "application/json"},
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
        req = urllib.request.Request(
            f"{self.api_url}/query",
            data=data,
            headers={**self.headers, "Content-Type": "application/json"},
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
        req = urllib.request.Request(
            f"{self.api_url}/detect",
            data=data,
            headers={**self.headers, "Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {"objects": result.get("objects", [])}
