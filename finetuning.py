# finetune_lora.py
import os, math, time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW

from model import SLM, SLMconfig
from lora import apply_lora_to_module, print_trainable_stats

data_dir = "data"
train_bin = os.path.join(data_dir, "gsm8k_train.bin")
val_bin   = os.path.join(data_dir, "gsm8k_val.bin")
out_dir = "out_lora"
os.makedirs(out_dir, exist_ok=True)

block_size = 256
batch_size = 4
grad_accum = 8
effective_batch = batch_size * grad_accum
max_iters = 4000
eval_interval = 250

# LoRA params
lora_r = 8
lora_alpha = 16
inject_mlp = False  
lr = 1e-4
weight_decay = 0.0
warmup_frac = 0.05

device = "cuda" if torch.cuda.is_available() else "cpu"

config = SLMconfig(
    vocab_size=50257,
    block_size=block_size,
    n_layer=4,
    n_head=6,
    n_embed=384,
    dropout=0.1
)
model = SLM(config).to(device)

ckpt = torch.load("out/ckpt.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])

use_qlora = False
try:
    import bitsandbytes as bnb
    use_qlora = False 
except Exception:
    use_qlora = False

apply_lora_to_module(model, r=lora_r, alpha=lora_alpha, inject_in_mlp=inject_mlp)
print_trainable_stats(model)

def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")

train_data = load_bin(train_bin)
val_data   = load_bin(val_bin)

def get_batch_from_memmap(data, batch_size, block_size):
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i:i+block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i+1:i+1+block_size].astype(np.int64) for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)

optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

warmup_steps = int(max_iters * warmup_frac)
def get_lr(step):
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, (max_iters - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    min_lr = lr * 0.1
    return min_lr + (lr - min_lr) * cosine

model.train()
global_step = 0
for step in range(max_iters):
    optimizer.zero_grad(set_to_none=True)
    cur_lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = cur_lr

    # gradient accumulation
    for _ in range(grad_accum):
        xb, yb = get_batch_from_memmap(train_data, batch_size, block_size)
        xb = xb.to(device)
        yb = yb.to(device)
        logits, loss = model(xb, yb)
        loss = loss / grad_accum
        loss.backward()

    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    global_step += 1

    if step % eval_interval == 0:
        # estimate val loss
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(100):
                xb, yb = get_batch_from_memmap(val_data, batch_size, block_size)
                xb = xb.to(device); yb = yb.to(device)
                _, vloss = model(xb, yb)
                val_losses.append(vloss.item())
            mean_val = sum(val_losses)/len(val_losses)
        model.train()
        print(f"[step {step}] lr={cur_lr:.2e} val_loss={mean_val:.4f}")

        # save only LoRA params
        lora_state = {k:v.cpu() for k,v in model.state_dict().items() if "lora_" in k or (v.requires_grad)}
        torch.save({"lora_state": lora_state, "config": config}, os.path.join(out_dir, f"lora_ckpt_{step}.pt"))

print("Fine-tuning complete.")
