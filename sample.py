from moondream import VisionEncoder, TextModel
from PIL import Image

vision_encoder = VisionEncoder()
text_model = TextModel()

image = Image.open("assets/demo-1.jpg")
image_embeds = vision_encoder(image)
out = text_model.generate(image_embeds, "User: <image>\nWhat is this?\nAssistant: ")

print(out)
