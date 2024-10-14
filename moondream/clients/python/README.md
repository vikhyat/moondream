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

```
pip install git+https://github.com/vikhyat/moondream.git#subdirectory=moondream/clients/python
```

Then download the model weights:

```
wget "https://huggingface.co/vikhyatk/moondream2/resolve/client/moondream-latest-f16.bin.gz?download=true" -O moondream-latest-f16.bin.gz
```

This downloads gzipped weights, which offers significant bandwidth savings.
You can load gzipped weights directly in the library, but to avoid runtime
decompression (which takes time), we recommend unzipping the weights:

```
gunzip moondream-latest-f16.bin.gz
```

## Usage

```python
import moondream as md
from PIL import Image

model = md.VL("moondream-latest-f16.bin")
image = Image.open("path/to/image.jpg")

# Optional -- encode the image to efficiently run multiple queries on the same
# image. This is not mandatory, since the model will automatically encode the
# image if it is not already encoded.
encoded_image = model.encode_image(image)

# Caption the image.
for t in model.caption(encoded_image, {"streaming": True}):
    print(t, end="", flush=True)
```
