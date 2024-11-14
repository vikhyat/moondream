import moondream as md

api_key = "XCP.LtHlcTTFXCgy6hJDKCds_XlT6An7fP9fao8qDMdNBNZRCwvAAds4W-TFtc_rdlSkCeLOcVIk9ZdSao-oKE-OUSW_Sy79wbBUmbwGe7Ha145nDL2tvzt7ixU0_yuKLQ"
# endpoint= "fail-enclave-2603"
# endpoint = "passenger-language-4124"

# model = md.VL(api_key=api_key, model_endpoint=endpoint)

# image_path = "/Users/caleb/Downloads/road-test.jpg"
# prompt = "What is the speed limit?"
# response = model.query(image_path, prompt)
# print(response)

endpoint = "hyacinth-ordination-9592"
model = md.Classifier(api_key=api_key, model_endpoint=endpoint)
image_path = "/Users/caleb/Projects/moondream/moondream-inf/test-scripts/out2/classification/hamburger/18221_565552128_173484.jpg"
response = model.classify(image_path)
print(response)
