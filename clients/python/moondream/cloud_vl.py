import base64
import json
import urllib.request
from io import BytesIO
from typing import Literal, Optional, Union

from PIL import Image

from .types import (
    VLM,
    Base64EncodedImage,
    CaptionOutput,
    DetectOutput,
    EncodedImage,
    PointOutput,
    QueryOutput,
    SamplingSettings,
)
from .version import __version__


class CloudVL(VLM):
    def __init__(
        self,
        *,
        api_url: str = "https://api.moondream.ai/v1",
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key
        self.api_url = api_url

    def encode_image(
        self, image: Union[Image.Image, EncodedImage]
    ) -> Base64EncodedImage:
        if isinstance(image, EncodedImage):
            assert type(image) == Base64EncodedImage
            return image
        try:
            width, height = image.size
            max_size = 768
            scale = max_size / max(width, height)
            if scale < 1:
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            if image.mode != "RGB":
                image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return Base64EncodedImage(image_url=f"data:image/jpeg;base64,{img_str}")
        except Exception as e:
            raise ValueError("Failed to convert image to JPEG.") from e

    def _stream_response(self, req):
        """Helper function to stream response chunks from the API."""
        with urllib.request.urlopen(req) as response:
            for line in response:
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "chunk" in data:
                            yield data["chunk"]
                        if data.get("completed"):
                            break
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            "Failed to parse JSON response from server."
                        ) from e

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: Literal["normal", "short"] = "normal",
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> CaptionOutput:
        encoded_image = self.encode_image(image)
        payload = {
            "image_url": encoded_image.image_url,
            "length": length,
            "stream": stream,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
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
            return {"caption": result["caption"]}

    def query(
        self,
        image: Union[Image.Image, EncodedImage],
        question: str,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> QueryOutput:
        encoded_image = self.encode_image(image)
        payload = {
            "image_url": encoded_image.image_url,
            "question": question,
            "stream": stream,
            # TODO: Pass sampling settings like max_tokens to the API.
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
        req = urllib.request.Request(
            f"{self.api_url}/query",
            data=data,
            headers=headers,
        )

        if stream:
            return {"answer": self._stream_response(req)}

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {"answer": result["answer"]}

    def detect(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
    ) -> DetectOutput:
        encoded_image = self.encode_image(image)
        payload = {"image_url": encoded_image.image_url, "object": object}

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
        req = urllib.request.Request(
            f"{self.api_url}/detect",
            data=data,
            headers=headers,
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {"objects": result["objects"]}

    def point(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
    ) -> PointOutput:
        encoded_image = self.encode_image(image)
        payload = {"image_url": encoded_image.image_url, "object": object}

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
        req = urllib.request.Request(
            f"{self.api_url}/point",
            data=data,
            headers=headers,
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {"points": result["points"]}
