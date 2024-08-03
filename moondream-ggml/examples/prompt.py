import moondream_ggml as moondream
import os

moondream.init(data_path='../data/', n_threads=8)
response = moondream.prompt(
    image_path=os.path.abspath('../assets/demo-1.jpg'), 
    prompt_str='<image>\n\nQuestion: Describe the image.\n\nAnswer:', 
    n_max_gen=128, 
    log_response_stream=True
)
moondream.cleanup()
