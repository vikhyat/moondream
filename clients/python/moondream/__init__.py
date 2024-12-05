from typing import Optional

from .cloud_vl import CloudVL
from .onnx_vl import OnnxVL
from .types import VLM

DEFAULT_API_URL = "https://api.moondream.ai/v1"


def vl(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = DEFAULT_API_URL,
) -> VLM:
    if model:
        return OnnxVL.from_path(model)

    if api_key:
        return CloudVL(api_key=api_key, api_url=api_url)

    if api_url:
        if api_url == DEFAULT_API_URL and not api_key:
            raise ValueError("An api_key is required for cloud inference.")

        return CloudVL(api_url=api_url)

    raise ValueError("At least one of `model`, `api_key`, or `api_url` is required.")
