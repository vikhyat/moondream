import os
from moondream_ggml_python.moondream_ggml_ffi import moondream_init_api_state 

TEXT_MODEL_FNAME = 'moondream2-text-model-f16.gguf'
MMPROJ_MODEL_FNAME = 'moondream2-mmproj-f16.gguf'

def init_api(data_path, num_threads):
    moondream_init_api_state(
        os.path.join(data_path, TEXT_MODEL_FNAME), 
        os.path.join(data_path, MMPROJ_MODEL_FNAME), 
        num_threads
    )
