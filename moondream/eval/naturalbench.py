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

# Yes, the benchmark test set is stored in the 'train' split...
dataset = load_dataset("BaiqiL/NaturalBench", split="train")

acc = []
q_acc = []
i_acc = []
g_acc = []

for row in tqdm(dataset):
    if row["Question_Type"] == "yes_no":
        suffix = " Answer yes or no."
    else:
        suffix = ""

    answers = moondream.batch_answer(
        images=[row["Image_0"], row["Image_1"], row["Image_0"], row["Image_1"]],
        prompts=[
            row["Question_0"] + suffix,
            row["Question_0"] + suffix,
            row["Question_1"] + suffix,
            row["Question_1"] + suffix,
        ],
        tokenizer=tokenizer,
    )

    expected = [
        row["Image_0_Question_0"],
        row["Image_1_Question_0"],
        row["Image_0_Question_1"],
        row["Image_1_Question_1"],
    ]

    acc.append(answers[0] == expected[0])
    acc.append(answers[1] == expected[1])
    acc.append(answers[2] == expected[2])
    acc.append(answers[3] == expected[3])

    i_acc.append(answers[0] == expected[0] and answers[2] == expected[2])
    i_acc.append(answers[1] == expected[1] and answers[3] == expected[3])

    q_acc.append(answers[0] == expected[0] and answers[1] == expected[1])
    q_acc.append(answers[2] == expected[2] and answers[3] == expected[3])

    g_acc.append(
        answers[0] == expected[0]
        and answers[1] == expected[1]
        and answers[2] == expected[2]
        and answers[3] == expected[3]
    )


print("Overall Accuracy:", sum(acc) / len(acc))
print("Image Accuracy:", sum(i_acc) / len(i_acc))
print("Question Accuracy:", sum(q_acc) / len(q_acc))
print("Group Accuracy:", sum(g_acc) / len(g_acc))
