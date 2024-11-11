import moondream as md

api_key = "XCP.rZtNgJVnJSU4G5ntI5GwAPF8KnnXdlL7jL0grajjabKd9quhSzyOhZhzsUUox_OdQN4WWQBKZvmy6YsNlysR8zUeLADJr_n9-Ze1rccq1gqPaT2GTepznOzSXOVuNQ"
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
