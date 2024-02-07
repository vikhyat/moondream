# 🌔 moondream

a tiny vision language model that kicks ass and runs anywhere

## moondream1

1.6B parameter model built using SigLIP, Phi-1.5 and the LLaVA training dataset.
Weights are licensed under CC-BY-SA due to using the LLaVA dataset. Try it out
on [Hugging Face Spaces](https://huggingface.co/spaces/vikhyatk/moondream1)!

**Benchmarks**

| Model | Parameters | VQAv2 | GQA | TextVQA |
| --- | --- | --- | --- | --- |
| LLaVA-1.5 | 13.3B | 80.0 | 63.3 | 61.3 |
| LLaVA-1.5 | 7.3B | 78.5 | 62.0 | 58.2 |
| **moondream1** | 1.6B | 74.7 | 57.9 | 35.6 |

**Examples**

| Image | Examples |
| --- | --- |
| ![](assets/demo-1.jpg) | **What is the title of this book?**<br>The Little Book of Deep Learning<br><br>**What can you tell me about this book?**<br>The book in the image is titled "The Little Book of Deep Learning." It appears to be a guide or manual that provides information and instructions on the subject of deep learning. The book is described as being small and thin, making it easy to carry and read. It is set in a room with a view outside, suggesting that the person holding the book might be in a living or working space with a view. The content of the book is likely to be complex, as the title implies a focus on advanced concepts in deep learning.<br><br>**Where is the person standing?**<br> The person is standing on a balcony or deck, which is outside. |
| ![](assets/demo-2.jpg) | **What type of food is the girl holding?**<br>The girl is holding a hamburger.<br><br>**What color is the woman's hair?**<br>The woman's hair is white.<br><br>**What is the girl doing in the image?**<br>The girl in the image is eating a hamburger. |
| ![](assets/demo-3.jpg) | **What color is the train?**<br>The train is red.<br><br>**What is the bus doing in the image?**<br> The bus in the image is driving down a city street, passing through an intersection.<br><br>**What kind of public transportation is in the image?**<br>The image features a red train or trolley as the public transportation option. |
| ![](assets/demo-4.jpg) | **What is the girl looking at?**<br>The girl is looking at her reflection in the mirror.<br><br>**What color is the woman's hair?**<br>The woman's hair is brown.<br><br>**What is the woman doing in the mirror?**<br>The woman is adjusting her hair in the mirror. |
| ![](assets/demo-5.jpg) | **What might be the reason for the dog's aggressive behavior?**<br>The dog's aggressive behavior, including its growling, baring its teeth, and showing its sharp teeth, could be due to various reasons. It is possible that the dog is feeling threatened, scared, or defensive about something in its environment, such as another animal, a loud noise, or a person nearby. Alternatively, the dog might be reacting to a command it received from its owner, or it could be experiencing discomfort or pain, which is causing it to display aggressive behavior. It is essential to observe the dog's body language and context to determine the underlying cause of its aggressive behavior and respond accordingly to address its needs or concerns.<br><br>**What kind of dog is in the picture?**<br>The dog in the picture is a small, black and white chihuahua. |

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
