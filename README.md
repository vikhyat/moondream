# 🌔 moondream

a tiny vision language model that kicks ass and runs anywhere

[Website](https://moondream.ai/) | [Hugging Face](https://huggingface.co/vikhyatk) | [Demo](https://huggingface.co/spaces/vikhyatk/moondream2)

![alt text](https://github.com/KPCOFGS/moondream_fork/blob/main/assets/sample_screenshot.png?raw=true)

## Benchmarks

moondream2 is a 1.86B parameter model initialized with weights from [SigLIP](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) and [Phi 1.5](https://huggingface.co/microsoft/phi-1_5).

| Model | VQAv2 | GQA | TextVQA | TallyQA (simple) | TallyQA (full) |
| --- | --- | --- | --- | --- | --- |
| moondream1 | 74.7 | 57.9 | 35.6 | - | - |
| **moondream2** (latest) | 77.7 | 61.7 | 49.7 | 80.1 | 74.2 |

## Downloading the model to local machine:
[moondream2 model](https://huggingface.co/vikhyatk/moondream2)
[moondream1 model](https://huggingface.co/vikhyatk/moondream1)
[moondream0 model](https://huggingface.co/vikhyatk/moondream0)

## Examples

| Image | Example |
| --- | --- |
| ![](assets/demo-1.jpg) | **What is the girl doing?**<br>The girl is sitting at a table and eating a large hamburger.<br><br>**What color is the girl's hair?**<br>The girl's hair is white. |
| ![](assets/demo-2.jpg) | **What is this?**<br>The image features a computer server rack, which is a large metal structure designed to hold and organize multiple computer components, such as motherboards, cooling systems, and other peripherals. The rack is filled with various computer parts, including multiple computer chips, wires, and other electronic components. The rack is placed on a carpeted floor, and there is a couch in the background, suggesting that the setup is likely in a living or working space.<br><br>**What is behind the stand?**<br>There is a brick wall behind the stand. |

## Usage

 [**Using transformers library**](usage/in_transformers.md) (recommended)

 [**Using from source code**](usage/in_source.md)


## Limitations

* The model may generate inaccurate statements, and struggle to understand intricate or nuanced instructions.
* The model may not be free from societal biases. Users should be aware of this and exercise caution and critical thinking when using the model.
* The model may generate offensive, inappropriate, or hurtful content if it is prompted to do so.

## License
This project is licensed under the Apache License

See the [LICENSE](LICENSE) file for details.
