# moondream

a tiny vision language model

## project goals

Build a high-quality, low-hallucination vision language model small enough to
run on an edge device without a GPU.

## moondream0

Initial prototype built using SigLIP, Phi-1.5, and the LLaVa training dataset.
The model is for research purposes only, and is subject to the Phi and LLaVa
license restrictions.

**Examples**

| Image | Examples |
| --- | --- |
| ![](assets/demo-1.jpg) | **What is the title of this book?**<br>The Little Book of Deep Learning<br>Author: Francois Flaubert<br><br>**What can you tell me about this book?**<br>The book in question is a small paperback titled "The Little Book of Deep Learning." It appears to be a guide or reference for those interested in deep learning, as the title suggests. The book is likely to be helpful for individuals who are new to the field or looking to expand their knowledge in this area. The image shows the book being held by a person, possibly a reader or a professional in the field.<br><br>**What is the person holding?**<br> The person is holding a book, which is titled "The Little Book of Deep Learning." |
| ![](assets/demo-2.jpg) | **What type of food is the girl holding?**<br>The girl is holding a sandwich.<br><br>**What color is the woman's hair?**<br>The woman's hair is white.<br><br>**What is the girl doing in the image?**<br>The girl in the image is eating a sandwich, specifically a hamburger, while sitting at a table. |
| ![](assets/demo-3.jpg) | **What color is the train?**<br>The train is red.<br><br>**What is the bus doing in the image?**<br>The bus in the image is traveling down a city street, passing by tall buildings and driving under a street light.<br><br>**What kind of public transportation is in the image?**<br>The image features a red light-rail train, which is a type of public transportation. |
| ![](assets/demo-4.jpg) | **What is the girl looking at?**<br>The girl is looking at her reflection in a mirror.<br><br>**What color is the woman's hair?**<br>The woman has black hair.<br><br>**What is the woman doing in the mirror?**<br>The woman is looking at herself in the mirror, possibly adjusting her hair or makeup. |
| ![](assets/demo-5.jpg) | **What might be the reason for the dog's aggressive behavior?**<br>The dog's aggressive behavior could be attributed to the fact that it is being bitten by a person, which is a form of animal abuse. The dog's eyes are red, and it appears to be growling, which are signs of fear or pain. The aggressive behavior might be a result of the dog's attempt to protect itself or its owner from the perceived threat. It is essential to address such behavior in dogs to ensure their well-being and prevent potential harm to both the dog and the person involved.<br><br>**What kind of dog is in the picture?**<br>The dog in the picture is a Chihuahua.<br><br>**What color is the dog?**<br>The dog is black and white. |
| ![](assets/demo-6.jpg) | **What is the man wearing?**<br>The man is wearing a large, inflatable suit or a giant inflatable costume.<br><br>**Is he indoors or outdoors?**<br>The man is outdoors, standing in front of a building.<br><br>**What is in the background?**<br>In the background, there is a building visible behind the person dressed up in a large amount of plastic wrap. |

**Usage**

Clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Use the `sample.py` script to run the model on CPU:

```bash
python sample.py --image [IMAGE_PATH] [--interactive]
```

When the `--interactive` flag is not set, the script will predict three questions and try
to answer them.

**Limitations**

* The model may generate inaccurate statements.
* It may struggle to adhere to intricate or nuanced instructions.
* It is primarily designed to understand English. Informal English, slang, and
  non-English languages may not work well.
* The model may not be free from societal biases. Users should be aware of this
  and exercise caution and critical thinking when using the model.
* The model may generate offensive, inappropriate, or hurtful content if it is
  prompted to do so.
