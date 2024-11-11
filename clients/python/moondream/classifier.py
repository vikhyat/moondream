from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from PIL import Image
from io import BytesIO
from moondream.utils import BASE_URL, validate_image, API_VERSION
import httpx
import uuid


class Classifier:
    """A client for making image classification requests to the Moondream API.
    Support for local inference is not yet supported, but is coming soon.

    Args:
        api_key (Optional[str]): The API key for authentication. Defaults to None.
        model_endpoint (Optional[str]): Custom model endpoint path. Defaults to None.
        model_path (Optional[str]): Local model path (not supported). Defaults to None.
    """

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
        self, image_buffer: BytesIO, classes: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """Makes an HTTP request to the classification endpoint.

        Args:
            image_buffer (BytesIO): The image data to be classified.

        Returns:
            List[Dict[str, Any]]: The raw response from the server.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        request_id = str(uuid.uuid4())
        request_files = {"content": (f"{request_id}.jpg", image_buffer, "image/jpeg")}

        request_headers = {
            "X-MD-Auth": self.api_key,
        }

        request_data = {}
        if classes:
            request_data = {"classes": classes}

        request_url = f"{BASE_URL}/{API_VERSION}/expert/classify"
        if self.model_endpoint:
            request_url = f"{BASE_URL}/{API_VERSION}/{self.model_endpoint}"

        response = self.httpx_client.post(
            request_url,
            files=request_files,
            headers=request_headers,
            data=request_data,
        )
        response.raise_for_status()

        return dict(response.json())

    def classify(
        self,
        image: Union[Image.Image, Path, str],
        classes: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Classifies the given image using the Moondream API.

        Args:
            image (Union[Image.Image, Path, str]): The image to classify. Can be a PIL Image,
                a path to an image file, or a base64 encoded image string.
            classes (Optional[List[Dict[str, str]]]): A list of dictionaries containing
                the class names and their corresponding labels, and descriptions.
                Should be in the format:
                    [{"label": str, "description": str}, ...]
                If you are using a distilled model, you can leave this blank.

        Returns:
            Dict[str, Any]: Classification result in the format...
                {"answer": [{"label": str, "confidence": float}]}. In descending order of confidence.
            If you are using an expert model, confidence scores will be 100.0 for the predicted class and 0.0 for the rest.
        """
        validated_image = validate_image(image)
        server_response = self._make_classification_request(validated_image, classes)
        return {"answer": server_response["result"]}
