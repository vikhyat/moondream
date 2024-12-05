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

Install the `CPU` library using pip

```bash
pip install moondream==0.0.2
```

Or the `GPU` library (requires [CUDA 12.x](https://docs.nvidia.com/cuda/) and [cuDNN 9.x](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html))

```bash
pip install moondream-gpu==0.0.2 # note: not deployed yet
```

Then download the model weights. We recommend using the int8 weights for most
applications, as they offer a good balance between memory usage and accuracy.

| Model          | Precision | Download Size | Memory Usage | Download Link                                                                                               |
| -------------- | --------- | ------------- | ------------ | ----------------------------------------------------------------------------------------------------------- |
| Moondream 2B   | int8      | 1,733 MiB     | 2,624 MiB    | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/onnx/moondream-2b-int8.mf.gz?download=true)   |
| Moondream 2B   | int4      | 1,167 MiB     | 2,002 MiB    | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/onnx/moondream-2b-int4.mf.gz?download=true)   |
| Moondream 0.5B | int8      | 593 MiB       | 996 MiB      | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/onnx/moondream-0_5b-int8.mf.gz?download=true) |
| Moondream 0.5B | int4      | 422 MiB       | 816 MiB      | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/onnx/moondream-0_5b-int4.mf.gz?download=true) |

While the library can load gzipped weights, we recommend decompressing the file
before usage to avoid paying the decompression cost every time the model is
initialized.

Memory usage is measured using `scripts/test.py`, and indicates the peak memory
usage expected during typical usage.

## Usage

```python
import moondream as md
from PIL import Image

model = md.vl(model="path/to/moondream-latest-int8.bin")
image = Image.open("path/to/image.jpg").convert("RGB")

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
