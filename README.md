# 🌔 moondream

a tiny vision language model that kicks ass and runs anywhere

[Website](https://moondream.ai/) | [Hugging Face](https://huggingface.co/vikhyatk/moondream2) | [Demo](https://huggingface.co/spaces/vikhyatk/moondream2)

## Benchmarks

moondream2 is a 1.86B parameter model initialized with weights from [SigLIP](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) and [Phi 1.5](https://huggingface.co/microsoft/phi-1_5).

| Model | VQAv2 | GQA | TextVQA | POPE | TallyQA |
| --- | --- | --- | --- | --- | --- |
| moondream1 | 74.7 | 57.9 | 35.6 | - | - |
| **moondream2** (latest) | 74.2 | 58.5 | 36.4 | (coming soon) | (coming soon) |

## Examples

| Image | Example |
| --- | --- |
| ![](assets/demo-1.jpg) | **What is the girl doing?**<br>The girl is sitting at a table, eating a burger.<br><br>**What color is the girl's hair?**<br>White |
| ![](assets/demo-2.jpg) | **What is this?**<br>A metal stand is positioned in the center of the image, with CPUs and wires visible. The background features a wall, and a black object is situated in the top left corner.<br><br>**What is behind the stand?**<br>A wall made of red bricks is visible behind the stand, which holds several electronic devices and wires. |

## Usage

**Using transformers** (recommended)

```bash
pip install transformers timm einops
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-03-05"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open('<IMAGE_PATH>')
enc_image = model.encode_image(image)
print(model.answer_question(enc_image, "Describe this image.", tokenizer))
```

The model is updated regularly, so we recommend pinning the model version to a
specific release as shown above.

**Using this repository**

Clone this repository and install dependencies.

```bash
pip install -r requirements.txt
```

`sample.py` provides a CLI interface for running the model. When the `--prompt` argument is not provided, the script will allow you to ask questions interactively.

```bash
python sample.py --image [IMAGE_PATH] --prompt [PROMPT]
```

Use `gradio_demo.py` script to start a Gradio interface for the model.

```bash
python gradio_demo.py
```

`webcam_gradio_demo.py` provides a Gradio interface for the model that uses your webcam as input and performs inference in real-time.

```bash
python webcam_gradio_demo.py
```

**Limitations**

* The model may generate inaccurate statements, and struggle to understand intricate or nuanced instructions.
* The model may not be free from societal biases. Users should be aware of this and exercise caution and critical thinking when using the model.
* The model may generate offensive, inappropriate, or hurtful content if it is prompted to do so.
