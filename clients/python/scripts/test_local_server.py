import moondream as md
from PIL import Image

base_url = "http://localhost:3475"
local_server_client = md.vl(api_url=base_url)

image_path = "../../assets/demo-1.jpg"
image = Image.open(image_path)

print("# Pointing")
object = "person"
print("Local Server:", local_server_client.point(image, object))

print("# Captioning")
print("Local Server:", local_server_client.caption(image))

print("# Querying")
question = "What is the character eating?"
print("Local Server:", local_server_client.query(image, question))

print("# Detecting")
object_to_detect = "burger"
print("Local Server:", local_server_client.detect(image, object_to_detect))

print("# Captioning Stream")
print("Local Server:")
for tok in local_server_client.caption(image, stream=True)["caption"]:
    print(tok, end="", flush=True)
print()

print("# Querying Stream")
print("Local Server:")
for tok in local_server_client.query(image, question, stream=True)["answer"]:
    print(tok, end="", flush=True)
print()
