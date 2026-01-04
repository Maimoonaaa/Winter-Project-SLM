import os
import math
import time
import torch
import numpy as np
from tqdm import tqdm

from model import SLM, SLMconfig

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 8
block_size = 256
gradient_accum_steps = 8
max_iters = 15000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

out_dir = "out"
os.makedirs(out_dir, exist_ok=True)

def load_data(split):
    data = np.memmap(
        f"data/{split}.bin",
        dtype=np.uint16,
        mode="r"
    )
    return data

train_data = load_data("train")
val_data   = load_data("test")

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i+block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64))
        for i in ix
    ])
    return x.to(device), y.to(device)

config = SLMconfig(
    vocab_size=50257,
    block_size=block_size,
    n_layer=4,
    n_head=6,
    n_embed=384,
    dropout=0.1
)

model = SLM(config).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.06
)
warmup_steps = int(0.05 * max_iters)
min_lr = learning_rate * 0.1

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(data)
            _, loss = model(X, Y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses

def get_lr(step):
    #linear warmup
    if step < warmup_steps:
        return learning_rate * step / warmup_steps

    #cosine decay
    progress = (step - warmup_steps) / (max_iters - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + cosine_decay * (learning_rate - min_lr)


print("Starting training...")
t0 = time.time()

for step in range(max_iters):
    optimizer.zero_grad(set_to_none=True)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    for _ in range(gradient_accum_steps):
        X, Y = get_batch(train_data)
        logits, loss = model(X, Y)
        loss = loss / gradient_accum_steps
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "config": config,
            "step": step
        }
        torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))


print("Training complete.")
