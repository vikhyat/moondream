import argparse
import os
import moondream as md

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
args = parser.parse_args()

local = md.vl(model_path=args.model_path)
cloud = md.vl(api_key=os.environ["MOONDREAM_API_KEY"])

image_path = "../../assets/demo-1.jpg"
image = Image.open(image_path)

print("# Pointing")
object = "person"
print("Local:", local.point(image, object))
print("Cloud:", cloud.point(image, object))

print("# Captioning")
print("Local:", local.caption(image))
print("Cloud:", cloud.caption(image))

print("# Querying")
question = "What is the character eating?"
print("Local:", local.query(image, question))
print("Cloud:", cloud.query(image, question))

print("# Detecting")
object_to_detect = "burger"
print("Local:", local.detect(image, object_to_detect))
print("Cloud:", cloud.detect(image, object_to_detect))

print("# Streaming Caption")
print("Local:")
for tok in local.caption(image, stream=True)["caption"]:
    print(tok, end="", flush=True)
print()
print("Cloud:")
for tok in cloud.caption(image, stream=True)["caption"]:
    print(tok, end="", flush=True)
print()

print("# Streaming Query")
print("Local:")
for tok in local.query(image, question, stream=True)["answer"]:
    print(tok, end="", flush=True)
print()
print("Cloud:")
for tok in cloud.query(image, question, stream=True)["answer"]:
    print(tok, end="", flush=True)
print()
