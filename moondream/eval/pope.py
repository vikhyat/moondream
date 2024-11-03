from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..hf import detect_device

MODEL_ID = "vikhyatk/moondream2"
DEVICE, DTYPE = detect_device()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
moondream = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)
moondream.eval()

pope_dataset = load_dataset("vikhyatk/POPE", split="test")

stats = {
    "random": (0, 0),
    "popular": (0, 0),
    "adversarial": (0, 0),
}
for row in tqdm(pope_dataset):
    image = row["image"]
    enc_image = moondream.encode_image(image)
    for split in ["adversarial", "popular", "random"]:
        for qa in row[split]:
            question = qa["question"]
            answer = qa["answer"]
            prompt = f"{question}\nAnswer yes or no."
            model_answer = moondream.answer_question(enc_image, prompt, tokenizer)
            if model_answer.lower() == answer.lower():
                stats[split] = (stats[split][0] + 1, stats[split][1] + 1)
            else:
                stats[split] = (stats[split][0], stats[split][1] + 1)

print(
    "Random:",
    stats["random"][0],
    "/",
    stats["random"][1],
    ":",
    stats["random"][0] * 100.0 / stats["random"][1],
)
print(
    "Popular:",
    stats["popular"][0],
    "/",
    stats["popular"][1],
    ":",
    stats["popular"][0] * 100.0 / stats["popular"][1],
)
print(
    "Adversarial:",
    stats["adversarial"][0],
    "/",
    stats["adversarial"][1],
    ":",
    stats["adversarial"][0] * 100.0 / stats["adversarial"][1],
)
