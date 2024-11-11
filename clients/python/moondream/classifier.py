from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from PIL import Image
from io import BytesIO
from moondream.utils import BASE_URL, validate_image
import httpx
import uuid


class Classifier:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_endpoint: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        if model_path:
            raise ValueError(
                "model_path is not yet supported for the classifier models"
            )

        self.api_key = api_key
        self.model_endpoint = model_endpoint
        self.httpx_client = httpx.Client(timeout=20.0)

    def _make_classification_request(
        self, image_buffer: BytesIO
    ) -> List[Dict[str, Any]]:
        request_id = str(uuid.uuid4())
        request_files = {"content": (f"{request_id}.jpg", image_buffer, "image/jpeg")}

        request_headers = {
            "X-MD-Auth": self.api_key,
        }

        request_url = f"{BASE_URL}/default"
        if self.model_endpoint:
            request_url = f"{BASE_URL}/{self.model_endpoint}"

        response = self.httpx_client.post(
            request_url,
            files=request_files,
            headers=request_headers,
        )
        response.raise_for_status()

        return dict(response.json())

    def classify(self, image: Union[Image.Image, Path, str]) -> Dict[str, Any]:
        validated_image = validate_image(image)
        server_response = self._make_classification_request(validated_image)
        return {"answer": server_response["result"]}
