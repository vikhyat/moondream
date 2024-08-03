import os
from moondream_ggml import cpp_ffi as ffi

TEXT_MODEL_FNAME = 'moondream2-text-model-f16.gguf'
MMPROJ_MODEL_FNAME = 'moondream2-mmproj-f16.gguf'

def init(data_path:str, n_threads:int) -> None:
    success = ffi.init(
        os.path.join(data_path, TEXT_MODEL_FNAME), 
        os.path.join(data_path, MMPROJ_MODEL_FNAME), 
        n_threads
    )
    if not success:
        raise RuntimeError()

def cleanup() -> None:
    ffi.cleanup()

def prompt(image_path:str, prompt_str:str, n_max_gen:int, log_response_stream:bool) -> str:
    success, response = ffi.prompt(image_path, prompt_str, n_max_gen, log_response_stream)
    if not success:
        raise RuntimeError()
    return response
