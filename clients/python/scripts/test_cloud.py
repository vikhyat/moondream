import moondream as md

api_key = "XCP.rZtNgJVnJSU4G5ntI5GwAPF8KnnXdlL7jL0grajjabKd9quhSzyOhZhzsUUox_OdQN4WWQBKZvmy6YsNlysR8zUeLADJr_n9-Ze1rccq1gqPaT2GTepznOzSXOVuNQ"
# endpoint= "fail-enclave-2603"
endpoint = "passenger-language-4124"

model = md.VL(api_key=api_key, model_endpoint=endpoint)

image_path = "/Users/caleb/Downloads/road-test.jpg"
prompt = "What is the speed limit?"
response = model.query(image_path, prompt)
print(response)


# model = md.Classifier(api_key=api_key, model_endpoint=endpoint)
# image_path = "/Users/caleb/Projects/moondream/moondream-inf/test-scripts/out2/classification/hamburger/18221_565552128_173484.jpg"
# response = model.classify(image_path)
# print(response)
