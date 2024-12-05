from typing import Optional
from .types import VLM
from .onnx_vl import OnnxVL
from .cloud_vl import CloudVL


def vl(*, model_path: Optional[str] = None, api_key: Optional[str] = None) -> VLM:
    if not model_path and not api_key:
        raise ValueError("Either model_path or api_key must be provided")

    if api_key:
        return CloudVL.from_api_key(api_key)

    return OnnxVL.from_path(model_path)
