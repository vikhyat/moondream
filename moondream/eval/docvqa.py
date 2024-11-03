import editdistance
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


def get_anls(s1, s2):
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
    anls = iou if iou >= 0.5 else 0.0
    return anls


docvqa_val = load_dataset("vikhyatk/docvqa", split="validation")

scores = []
for row in tqdm(docvqa_val):
    image = row["image"]
    enc_image = moondream.encode_image(image)
    for qa in row["qa"]:
        question = qa["question"]
        answers = qa["answers"]
        prompt = f"{question}\nAnswer briefly with a single word or phrase."

        model_answer = moondream.answer_question(enc_image, prompt, tokenizer)
        anls = max(get_anls(model_answer, gt) for gt in answers)
        scores.append(anls)

print("ANLS:", sum(scores) / len(scores))
