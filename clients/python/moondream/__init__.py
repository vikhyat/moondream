from .types import VLM
from .onnx_vl import OnnxVL
from .cloud_vl import CloudVL


def VL(
    model_path: str = None,
    api_key: str = None,
    api_url: str = None,
    version: str = "v1",
) -> VLM:
    if not model_path and not api_key:
        raise ValueError("Either model_path or api_key must be provided")

    if api_key:
        return CloudVL.from_api_key(api_key, api_url, version)

    return OnnxVL.from_path(model_path)
