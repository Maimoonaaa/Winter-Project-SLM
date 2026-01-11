import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

SUBSET_PERCENT = 1
VAL_FRACTION = 0.01

out_dir = "data"
os.makedirs(out_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")

TOTAL_EXAMPLES = 8_000_000
NUM_SAMPLES = int(TOTAL_EXAMPLES * SUBSET_PERCENT / 100)
NUM_VAL = int(NUM_SAMPLES * VAL_FRACTION)
NUM_TRAIN = NUM_SAMPLES - NUM_VAL

print(f"Loading {NUM_SAMPLES:,} samples (streaming)")

dataset = load_dataset(
    "openwebtext",
    split="train",
    streaming=True
)

dataset = dataset.take(NUM_SAMPLES)

train_tokens = []
val_tokens = []

def encode(text):
    ids = enc.encode(text)
    ids.append(enc.eot_token)
    return ids

print("Tokenizing & splitting")

for i, ex in enumerate(tqdm(dataset, total=NUM_SAMPLES)):
    ids = encode(ex["text"])
    if i < NUM_VAL:
        val_tokens.extend(ids)
    else:
        train_tokens.extend(ids)

# ---- write binaries ----
def write_bin(path, tokens):
    arr = np.memmap(
        path,
        dtype=np.uint16,
        mode="w+",
        shape=(len(tokens),)
    )
    arr[:] = np.array(tokens, dtype=np.uint16)
    arr.flush()

print(f"Writing train.bin ({len(train_tokens):,} tokens)")
write_bin(os.path.join(out_dir, "train.bin"), train_tokens)

print(f"Writing test.bin ({len(val_tokens):,} tokens)")
write_bin(os.path.join(out_dir, "test.bin"), val_tokens)

print("Dataset preparation complete.")
