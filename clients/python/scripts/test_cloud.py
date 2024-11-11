import moondream as md

api_key = "..."
# endpoint= "fail-enclave-2603"

# model = md.VL(api_key=api_key, model_endpoint=endpoint)

# image_path = "/Users/caleb/Projects/moondream/moondream-inf/test-scripts/out2/classification/hamburger/18221_565552128_173484.jpg"
# prompt = "Is this a hotdog or a hamburger?"
# response = model.query(image_path, prompt)
# print(response)


endpoint = "frog-rear-9189"

model = md.Classifier(api_key=api_key, model_endpoint=endpoint)
image_path = "/Users/caleb/Projects/moondream/moondream-inf/test-scripts/out2/classification/hamburger/18221_565552128_173484.jpg"
response = model.classify(image_path)
print(response)
