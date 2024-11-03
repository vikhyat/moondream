# Expects Visual Genome to be downloaded to `data/vg` and the TallyQA test set
# to be present at `data/tallyqa/test.json`.
#
# Steps to download Visual Genome and TallyQA:
#
#   mkdir -p data/vg/VG_100K
#   mkdir -p data/vg/VG_100K_2
#   mkdir -p data/tallyqa
#   wget -P data/vg/VG_100K_2/ https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
#   wget -P data/vg/VG_100K/ https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
#   wget -P data/tallyqa/ https://github.com/manoja328/TallyQA_dataset/raw/master/tallyqa.zip
#   unzip data/vg/VG_100K_2/images2.zip -d data/vg/
#   unzip data/vg/VG_100K/images.zip -d data/vg/
#   unzip data/tallyqa/tallyqa.zip -d data/tallyqa/
#   rm data/vg/VG_100K_2/images2.zip
#   rm data/vg/VG_100K/images.zip
#   rm data/tallyqa/tallyqa.zip

import json

from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from ..hf import Moondream, detect_device

BATCH_SIZE = 16
DEVICE, DTYPE = detect_device()

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = Moondream.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)
model.eval()

total = 0
total_simple = 0
correct = 0
correct_simple = 0

# Iterate over tallyqa_test in batches of BATCH_SIZE
tallyqa_test = json.load(open("data/tallyqa/test.json"))
for i in tqdm(range(0, len(tallyqa_test), BATCH_SIZE)):
    batch = tallyqa_test[i : i + BATCH_SIZE]

    images = [Image.open(f"data/vg/{item['image']}") for item in batch]
    questions = [
        item["question"] + " Answer in a word or phrase only." for item in batch
    ]

    answers = model.batch_answer(
        images=images, prompts=questions, tokenizer=tokenizer, max_new_tokens=10
    )

    for answer, item in zip(answers, batch):
        is_simple = item["issimple"]
        is_correct = 1 if str(item["answer"]) == answer else 0

        total += 1
        correct += is_correct
        if is_simple:
            total_simple += 1
            correct_simple += is_correct

print(
    f"Simple: {total_simple}, Correct: {correct_simple}, Accuracy: {correct_simple*100.0/total_simple}"
)
print(f"Total: {total}, Correct: {correct}, Accuracy: {correct*100.0/total}")
