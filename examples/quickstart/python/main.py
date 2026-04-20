import moondream as md
from PIL import Image

# This will run the model locally
model = md.vl(endpoint="http://localhost:2020/v1")

# For Moondream Cloud, use your API key:
# model = md.vl(api_key="<your-api-key>")

# Load an image
image = Image.open("../../images/frieren.jpg")

# Example: Generate a caption
caption_response = model.caption(image, length="short")
print(caption_response["caption"])
