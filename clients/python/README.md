# Moondream Python Client Library

Python client library for moondream. This library is an alpha preview -- it is
in an early stage of development, and backward compatibility is not yet
guaranteed. If you are using this in production, please pin the revision you
are using.

This library currently offers optimized CPU inference, but will be slower than
the PyTorch implementation for CUDA and MPS backends. If you are running on a
Mac with M1/M2/M3 etc. chips, or if you have a GPU available, this library is
not recommended yet.

## Setup

Install the library using pip:

```bash
pip install moondream==0.0.2
```

Then download the model weights:

```bash
# int8 weights (recommended):
wget "https://huggingface.co/vikhyatk/moondream2/resolve/client/moondream-latest-int8.bin.gz?download=true" -O - | gunzip > moondream-latest-int8.bin
# ...or, for fp16 weights (full precision):
wget "https://huggingface.co/vikhyatk/moondream2/resolve/client/moondream-latest-f16.bin.gz?download=true" -O - | gunzip > moondream-latest-f16.bin
# ...or, for int4 weights (resource constrained environments):
wget "https://huggingface.co/vikhyatk/moondream2/resolve/client/moondream-latest-int4.bin.gz?download=true" -O - | gunzip > moondream-latest-int4.bin
```

## Usage

```python
import moondream as md
from PIL import Image

model = md.VL("moondream-latest-int8.bin")
image = Image.open("path/to/image.jpg")

# Optional -- encode the image to efficiently run multiple queries on the same
# image. This is not mandatory, since the model will automatically encode the
# image if it is not already encoded.
encoded_image = model.encode_image(image)

# Caption the image.
caption = model.caption(encoded_image)

# ...or, if you want to stream the output:
for t in model.caption(encoded_image, stream=True)["caption"]:
    print(t, end="", flush=True)

# Ask a question about the image.
question = "How many people are in this image?"
answer = model.query(encoded_image, question)["answer"]

# ...or again, if you want to stream the output:
for t in model.query(encoded_image, question, stream=True)["answer"]:
    print(t, end="", flush=True)
```
