from PIL import Image
from transformers import AutoTokenizer

from moondream.hf import LATEST_REVISION, Moondream, detect_device

device, dtype = detect_device()

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = Moondream.from_pretrained(
    model_id,
    revision=LATEST_REVISION,
    torch_dtype=dtype,
).to(device=device)
moondream.eval()

image1 = Image.open("assets/demo-1.jpg")
image2 = Image.open("assets/demo-2.jpg")
prompts = [
    "What is the girl doing?",
    "What color is the girl's hair?",
    "What is this?",
    "What is behind the stand?",
]

answers = moondream.batch_answer(
    images=[image1, image1, image2, image2],
    prompts=prompts,
    tokenizer=tokenizer,
)

for question, answer in zip(prompts, answers):
    print(f"Q: {question}")
    print(f"A: {answer}")
    print()
