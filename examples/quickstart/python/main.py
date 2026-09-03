import moondream as md
from PIL import Image
import os

# This will run the model locally
model = md.vl(endpoint="http://localhost:2020/v1")

# For Moondream Cloud, use your API key from the MOONDREAM_API_KEY
# environment variable instead of hardcoding it:
# model = md.vl(api_key=os.environ.get("MOONDREAM_API_KEY"))

# Load an image
image = Image.open("../../images/frieren.jpg")

# Example: Generate a caption
caption_response = model.caption(image, length="short")
print(caption_response["caption"])
