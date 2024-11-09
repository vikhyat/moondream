import moondream as md

api_key = "..."
endpoint = "..."

model = md.VL(api_key=api_key, model_endpoint=endpoint)

image_path = "/Users/caleb/Projects/moondream/moondream-inf/test-scripts/out2/classification/hamburger/18221_565552128_173484.jpg"
prompt = "Is this a hotdog or a hamburger?"
response = model.query(image_path, prompt)
print(response)
