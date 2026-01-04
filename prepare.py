import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

SUBSET_PERCENT = 0.3    # change to 2 for 2%
VAL_FRACTION = 0.01      # 1% of subset used for validation
num_proc = 4 

out_dir = "data"
os.makedirs(out_dir, exist_ok=True)
#kdk
enc = tiktoken.get_encoding("gpt2")

print(f"Loading data")
dataset = load_dataset(
    "openwebtext",
    split=f"train[:{SUBSET_PERCENT}%]"
)

split_dataset = dataset.train_test_split(
    test_size=VAL_FRACTION,
    seed=2357,
    shuffle=True
)

def process(example):
    ids = enc.encode(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}

tokenized = split_dataset.map(
    process,
    remove_columns=["text"],
    desc="Tokenizing",
    num_proc=num_proc
)

for split, data in tokenized.items():
    arr_len = np.sum(data["len"], dtype=np.uint64)
    filename = os.path.join(out_dir, f"{split}.bin")

    print(f"Writing {filename} ({arr_len:,} tokens)")
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

    idx = 0
    for batch in tqdm(data, desc=f"Writing {split}"):
        arr[idx : idx + batch["len"]] = np.array(batch["ids"], dtype=dtype)
        idx += batch["len"]

    arr.flush()

print("Dataset preparation complete.")
