from typing import Optional

from .cloud_vl import CloudVL
from .types import VLM

DEFAULT_API_URL = "https://api.moondream.ai/v1"


def vl(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> VLM:
    if model:
        model_filetype = model.split(".")[-1]
        if model_filetype == "safetensors":
            from .torch_vl import TorchVL

            return TorchVL(model=model)
        elif model_filetype == "mf":
            from .onnx_vl import OnnxVL

            return OnnxVL.from_path(model)

        raise ValueError(
            "Unsupported model filetype. Please use a .safetensors model for GPU use or .mf model for CPU use."
        )

    if api_key:
        if not api_url:
            api_url = DEFAULT_API_URL

        return CloudVL(api_key=api_key, api_url=api_url)

    if api_url and api_url == DEFAULT_API_URL:
        if not api_key:
            raise ValueError("An api_key is required for cloud inference.")

        return CloudVL(api_url=api_url)

    raise ValueError("At least one of `model`, `api_key`, or `api_url` is required.")
