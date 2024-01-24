# 🌔 moondream

a tiny vision language model that kicks ass and runs anywhere

## moondream1

1.6B parameter model built using SigLIP, Phi-1.5 and the LLaVA training dataset.
Weights are licensed under CC-BY-SA due to using the LLaVA dataset. Try it out
on [Huggingface Spaces](https://huggingface.co/spaces/vikhyatk/moondream1)!

**Benchmarks**

| Model | Parameters | VQAv2 | GQA | VizWiz | TextVQA |
| --- | --- | --- | --- | --- | --- |
| LLaVA-1.5 | 13.3B | 80.0 | 63.3 | 53.6 | 61.3 |
| LLaVA-1.5 | 7.3B | 78.5 | 62.0 | 50.0 | 58.2 |
| [MC-LLaVA-3B](https://huggingface.co/visheratin/MC-LLaVA-3b) | 3B | 64.2 | 49.6 | 24.9 | 38.6 |
| [LLaVA-Phi](https://arxiv.org/pdf/2401.02330.pdf) | 3B | 71.4 | - | 35.9 | 48.6 |
| **moondream1** | 1.6B | 74.3 | 56.3 | 30.3 | 39.8 |

**Examples**

| Image | Examples |
| --- | --- |
| ![](assets/demo-1.jpg) | **What is the title of this book?**<br>The Little Book of Deep Learning<br><br>**What can you tell me about this book?**<br>The book appears to be a white booklet titled "The Little Book of Deep Learning." It is held in a person's hand, and it seems to be a personal possession. The book's content focuses on the basics of deep learning, which is a field of artificial intelligence that uses neural networks to process and analyze data. It is likely that the book provides an introduction to the concepts and techniques involved in deep learning, making it accessible for beginners and helping them understand the fundamentals of this advanced machine learning approach.<br><br>**Where is the person standing?**<br>The person is standing on a balcony or a deck, which is located outside the house. |
| ![](assets/demo-2.jpg) | **What type of food is the girl holding?**<br>The girl is holding a large hamburger or burger, which is a type of sandwich made from ground meat, typically consisting of a beef patty, and is usually served between two slices of bread.<br><br>**What color is the woman's hair?**<br>The woman's hair is white.<br><br>**What is the girl doing in the image?**<br>The girl in the image is eating a hamburger. |
| ![](assets/demo-3.jpg) | **What color is the train?**<br>The train is red.<br><br>**What is the bus doing in the image?**<br>The bus is driving down a street, passing through an intersection, and traveling on a train track.<br><br>**What kind of public transportation is in the image?**<br>The image features a red trolley or commuter train on a city street, which is a form of public transportation. |
| ![](assets/demo-4.jpg) | **What is the girl looking at?**<br>The girl is looking at her reflection in the mirror while adjusting her uniform.<br><br>**What color is the woman's hair?**<br>The woman's hair is brown.<br><br>**What is the woman doing in the mirror?**<br>The woman is adjusting her hair in the mirror. |
| ![](assets/demo-5.jpg) | **What might be the reason for the dog's aggressive behavior?**<br>The dog's aggressive behavior, with its teeth bared and growling, could be due to several reasons. It is possible that the dog is feeling threatened, scared, or defensive in its current environment, such as a room with a person it doesn't know well or a situation that provokes it. Alternatively, the dog might be reacting to a perceived threat or discomfort from the person holding it. It is essential to assess the situation and the dog's body language to determine the exact cause of its aggressive behavior and respond accordingly to ensure the safety and well-being of both the dog and the person involved.<br><br>**What kind of dog is in the picture?**<br>The picture features a small dog, possibly a Chihuahua, with red eyes and a mean, hungry-looking expression.<br><br>**What color is the dog?**<br>The dog is black and white. |

**Usage**

Clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Use the `sample.py` script to run the model on CPU:

```bash
python sample.py --image [IMAGE_PATH] --prompt [PROMPT]
```

When the `--prompt` argument is not provided, the script will allow you to ask
questions interactively.

**Gradio demo**

Use the `gradio_demo.py` script to run the gradio app:

```python
python gradio_demo.py
```


**Limitations**

* The model may generate inaccurate statements.
* It may struggle to adhere to intricate or nuanced instructions.
* It is primarily designed to understand English. Informal English, slang, and
  non-English languages may not work well.
* The model may not be free from societal biases. Users should be aware of this
  and exercise caution and critical thinking when using the model.
* The model may generate offensive, inappropriate, or hurtful content if it is
  prompted to do so.
