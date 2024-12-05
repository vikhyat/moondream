# Moondream Python Client Library

Official Python client library for Moondream, a tiny vision language model that
can analyze images and answer questions about them. This library supports both
local inference and cloud-based API access.

## Features

- **Local Inference**: Run the model directly on your machine using CPU
- **Cloud API**: Access Moondream's hosted service for faster inference
- **Streaming**: Stream responses token by token for real-time output
- **Multiple Model Sizes**: Choose between 0.5B and 2B parameter models
- **Multiple Tasks**: Caption images, answer questions, detect objects, and locate points

## Installation

Install the package from PyPI:

```bash
pip install moondream==0.0.5
```

## Quick Start

### Using Cloud API

To use Moondream's cloud API, you'll first need an API key. Sign up for a free
account at [console.moondream.ai](https://console.moondream.ai) to get your key.
Once you have your key, you can use it to initialize the client as shown below.

```python
import moondream as md
from PIL import Image

# Initialize with API key
model = md.vl(api_key="your-api-key")

# Load an image
image = Image.open("path/to/image.jpg")

# Generate a caption
caption = model.caption(image)["caption"]
print("Caption:", caption)

# Ask a question
answer = model.query(image, "What's in this image?")["answer"]
print("Answer:", answer)

# Stream the response
for chunk in model.caption(image, stream=True)["caption"]:
    print(chunk, end="", flush=True)
```

### Using Local Inference

First, download the model weights. We recommend the int8 weights for most applications:

| Model          | Precision | Download Size | Memory Usage | Download Link                                                                                                                                   |
| -------------- | --------- | ------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Moondream 2B   | int8      | 1,733 MiB     | 2,624 MiB    | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz?download=true)   |
| Moondream 2B   | int4      | 1,167 MiB     | 2,002 MiB    | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int4.mf.gz?download=true)   |
| Moondream 0.5B | int8      | 593 MiB       | 996 MiB      | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz?download=true) |
| Moondream 0.5B | int4      | 422 MiB       | 816 MiB      | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int4.mf.gz?download=true) |

Then use the model locally:

```python
import moondream as md
from PIL import Image

# Initialize with local model path
model = md.vl(model="path/to/moondream-2b-int8.bin")

# Load and encode image
image = Image.open("path/to/image.jpg")

# Since encoding an image is computationally expensive, you can encode it once
# and reuse the encoded version for multiple queries/captions/etc. This avoids
# having to re-encode the same image multiple times.
encoded_image = model.encode_image(image)

# Generate caption
caption = model.caption(encoded_image)["caption"]
print("Caption:", caption)

# Ask questions
answer = model.query(encoded_image, "What's in this image?")["answer"]
print("Answer:", answer)
```

## API Reference

### Constructor

```python
model = md.vl(
    model="path/to/model.bin",  # For local inference
    api_key="your-api-key"      # For cloud API access
)
```

### Methods

#### caption(image, length="normal", stream=False, settings=None)

Generate a caption for an image.

```python
result = model.caption(image)
# or with streaming
for chunk in model.caption(image, stream=True)["caption"]:
    print(chunk, end="")
```

#### query(image, question, stream=False, settings=None)

Ask a question about an image.

```python
result = model.query(image, "What's in this image?")
# or with streaming
for chunk in model.query(image, "What's in this image?", stream=True)["answer"]:
    print(chunk, end="")
```

#### detect(image, object)

Detect and locate specific objects in an image.

```python
result = model.detect(image, "car")
```

#### point(image, object)

Get coordinates of specific objects in an image.

```python
result = model.point(image, "person")
```

### Input Types

- Images can be provided as:
  - PIL.Image.Image objects
  - Encoded image objects (from model.encode_image())

### Response Types

All methods return typed dictionaries:

- CaptionOutput: `{"caption": str | Generator}`
- QueryOutput: `{"answer": str | Generator}`
- DetectOutput: `{"objects": List[Region]}`
- PointOutput: `{"points": List[Point]}`

## Performance Notes

- Local inference currently only supports CPU execution
- CUDA (GPU) and MPS (Apple Silicon) support coming soon
- For optimal performance with GPU/MPS, use the PyTorch implementation for now

## Links

- [Website](https://moondream.ai/)
- [Demo](https://moondream.ai/playground)
