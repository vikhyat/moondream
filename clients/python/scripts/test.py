import time

from PIL import Image

import moondream as md
from moondream.preprocess import create_patches

image = Image.open("../../assets/demo-1.jpg")

start_time = time.time()
model = md.VL("../../onnx/out/moondream-latest-int4.bin")
end_time = time.time()
print(f"Time to initialize model: {end_time - start_time:.2f} seconds")

start_time = time.time()
encoded_image = model.encode_image(image)
end_time = time.time()
print(f"Time to encode image: {end_time - start_time:.2f} seconds")
print()

print("Caption:", end="", flush=True)
start_time = time.time()
tokens = 0
for tok in model.caption(encoded_image, stream=True)["caption"]:
    print(tok, end="", flush=True)
    tokens += 1
print()
end_time = time.time()
print(
    f"Time to generate caption: {end_time - start_time:.2f} seconds"
    f" ({tokens / (end_time - start_time):.2f} tok/s)"
)
print()

question = "How many people are in this image?"
print(f"Question: {question}\nAnswer:", end="", flush=True)
start_time = time.time()
tokens = 0
for tok in model.query(encoded_image, question, stream=True)["answer"]:
    print(tok, end="", flush=True)
    tokens += 1
print()
end_time = time.time()
print(
    f"Time to generate answer: {end_time - start_time:.2f} seconds"
    f" ({tokens / (end_time - start_time):.2f} tok/s)"
)
