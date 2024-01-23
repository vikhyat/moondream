import re
import gradio as gr
from moondream import VisionEncoder, TextModel
from PIL import Image
from huggingface_hub import snapshot_download
from threading import Thread
from transformers import TextIteratorStreamer

model_path = snapshot_download("vikhyatk/moondream1")

vision_encoder = VisionEncoder(model_path)
text_model = TextModel(model_path)

# model inference
def moondream(img, prompt):
    
    image_embeds = vision_encoder(img)

    streamer = TextIteratorStreamer(text_model.tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(
        image_embeds=image_embeds, question=prompt, streamer=streamer
    )
    thread = Thread(target=text_model.answer_question, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        # check for the end of generated text and yield the generated token
        if not new_text.endswith("<") and not new_text.endswith("END"):
          buffer += new_text
          yield buffer
        else:
          new_text = re.sub("<$", "", re.sub("END$", "", new_text))
          buffer += new_text
          yield buffer

# Using Gradio Blocks API
with gr.Blocks() as demo:
  gr.HTML("<h1><center>🌔 moondream</center></h1>")
  gr.HTML("<h3><center>A tiny vision language model. <a href='https://github.com/vikhyat/moondream' target='blank_'>GitHub</a></center></h3>")
  with gr.Group():
    with gr.Row():
      prompt = gr.Textbox(label='Input Prompt for the model',placeholder='Type whatever you want to ask about the image',scale=4 )
      submit = gr.Button('Submit', scale=1,)
    with gr.Row():
      img = gr.Image(type='pil', label='Upload or Drag an Image')
      output = gr.TextArea(label="Bot's response to the user query-", info='The response might take a few seconds..' )
  
  # handling events
  submit.click(moondream, [img, prompt], output)
  prompt.submit(moondream, [img, prompt], output)

# launch gradio demo with debug mode on
demo.queue().launch(debug=True)
