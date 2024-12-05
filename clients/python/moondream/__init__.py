from typing import Optional
from .types import VLM
from .onnx_vl import OnnxVL
from .cloud_vl import CloudVL


def vl(*, model_path: Optional[str] = None, api_key: Optional[str] = None) -> VLM:
    if api_key:
        return CloudVL(api_key)

    if model_path:
        return OnnxVL.from_path(model_path)

    raise ValueError("Either model_path or api_key must be provided.")
