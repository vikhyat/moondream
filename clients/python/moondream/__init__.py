from .types import VLM
from .onnx_vl import OnnxVL

def VL(model_path: str) -> VLM:
    return OnnxVL.from_path(model_path)
