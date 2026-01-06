import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "data-FT"
os.makedirs(OUT_DIR, exist_ok=True)

VAL_FRACTION = 0.05
MAX_SAMPLES = None
enc = tiktoken.get_encoding("gpt2")

def format_gsm8k(example):
    q = example["question"].strip()
    a = example["answer"].strip()

    text = (
        "### Question:\n"
        f"{q}\n\n"
        "### Answer:\n"
        f"{a}\n\n"
        "### Final Answer:\n"
        f"{a.split('####')[-1].strip()}"
    )
    return text


def tokenize_and_write(texts, filename):
    all_ids = []
    for t in tqdm(texts):
        ids = enc.encode(t)
        ids.append(enc.eot_token)
        all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.uint16)
    memmap = np.memmap(filename, dtype=np.uint16, mode="w+", shape=arr.shape)
    memmap[:] = arr[:]
    memmap.flush()


print("Loading GSM8K...")
dataset = load_dataset("gsm8k", "main")

train_texts = []
val_texts = []

for ex in dataset["train"]:
    train_texts.append(format_gsm8k(ex))

for ex in dataset["test"]:
    val_texts.append(format_gsm8k(ex))

if MAX_SAMPLES:
    train_texts = train_texts[:MAX_SAMPLES]

print("Writing GSM8K bins...")
tokenize_and_write(train_texts, os.path.join(OUT_DIR, "gsm8k_train.bin"))
tokenize_and_write(val_texts,   os.path.join(OUT_DIR, "gsm8k_val.bin"))

print("GSM8K done.")
