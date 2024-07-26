import moondream_ggml as moondream

moondream.init('../../data/', 8)
response = moondream.prompt('<image>\n\nQuestion: Describe the image.\n\nAnswer:', 128, True)
moondream.cleanup()
