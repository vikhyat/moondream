from typing import Optional
from .types import VLM
from .onnx_vl import OnnxVL
from .cloud_vl import CloudVL


def vl(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = "https://api.moondream.ai/v1",
) -> VLM:
    if api_key:
        return CloudVL(api_key, api_url)

    if model:
        return OnnxVL.from_path(model)

    if api_url:
        return CloudVL(api_url=api_url)

    raise ValueError(
        "A model path is required for local inference. "
        "An api_key is required for cloud inference. "
        "An api_url is required for using the local server."
    )
