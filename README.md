# 🌔 moondream

a tiny vision language model that kicks ass and runs anywhere

[Website](https://moondream.ai/) | [Demo](https://moondream.ai/playground)

## Examples

| Image                  | Example                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![](assets/demo-1.jpg) | **What is the girl doing?**<br>The girl is sitting at a table and eating a large hamburger.<br><br>**What color is the girl's hair?**<br>The girl's hair is white.                                                                                                                                                                                                                                                                                                                                                                                                    |
| ![](assets/demo-2.jpg) | **What is this?**<br>This is a computer server rack, which is a device used to store and manage multiple computer servers. The rack is filled with various computer servers, each with their own dedicated space and power supply. The servers are connected to the rack via multiple cables, indicating that they are part of a larger system. The rack is placed on a carpeted floor, and there is a couch nearby, suggesting that the setup is in a living or entertainment area.<br><br>**What is behind the stand?**<br>Behind the stand, there is a brick wall. |

## About

Moondream is a highly efficient open-source vision language model that combines powerful image understanding capabilities with a remarkably small footprint. It's designed to be versatile and accessible, capable of running on a wide range of devices and platforms.

The project offers two model variants:

- **Moondream 2B**: The primary model with 2 billion parameters, offering robust performance for general-purpose image understanding tasks including captioning, visual question answering, and object detection.
- **Moondream 0.5B**: A compact 500 million parameter model specifically optimized as a distillation target for edge devices, enabling efficient deployment on resource-constrained hardware while maintaining impressive capabilities.

## Getting Started

### Latest Model Checkpoints

These are the latest bleeding-edge versions of both models, with all new features and improvements:

| Model          | Precision | Download Size | Memory Usage | Best For                   | Download Link                                                                                                                                   |
| -------------- | --------- | ------------- | ------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Moondream 2B   | int8      | 1,733 MiB     | 2,624 MiB    | General use, best quality  | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz?download=true)   |
| Moondream 0.5B | int8      | 593 MiB       | 996 MiB      | Edge devices, faster speed | [Download](https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz?download=true) |

### Python Client Library

First, install the client library:

```bash
pip install moondream==0.0.5
```

The recommended way to use the latest version of Moondream is through our Python client library:

```python
import moondream as md
from PIL import Image

# Initialize with local model path. Can also read .mf.gz files, but we recommend decompressing
# up-front to avoid decompression overhead every time the model is initialized.
model = md.vl(model="path/to/moondream-2b-int8.mf")

# Load and process image
image = Image.open("path/to/image.jpg")
encoded_image = model.encode_image(image)

# Generate caption
caption = model.caption(encoded_image)["caption"]
print("Caption:", caption)

# Ask questions
answer = model.query(encoded_image, "What's in this image?")["answer"]
print("Answer:", answer)
```

⚠️ Note: The Python client currently only supports CPU inference. CUDA (GPU) and MPS (Apple Silicon) optimization is coming soon. For GPU support, use the Hugging Face transformers implementation below.

For complete documentation of the Python client, including cloud API usage and additional features, see the [Python Client README](clients/python/README.md).

### Node.js Client Library

For JavaScript/TypeScript developers, we offer a full-featured Node.js client library. See the [Node.js Client README](https://github.com/rohan-kulkarni-25/moondream/blob/main/clients/node/README.MD) for installation and usage instructions.

### Hugging Face Transformers Integration

The Hugging Face hub version tracks the last official release of the 2B model. While more stable, it doesn't include the latest features or support for the 0.5B model. Use this if you need GPU acceleration or prefer the transformers ecosystem:

First, install the required packages:

```bash
pip install transformers torch einops
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"  # Pin to specific version
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open('<IMAGE_PATH>')
enc_image = model.encode_image(image)
print(model.answer_question(enc_image, "Describe this image.", tokenizer))
```

For GPU acceleration, you can add:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16, attn_implementation="flash_attention_2"
).to("cuda")
```
