# lora.py
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r=8, alpha=16, merge_weights=False):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.bias = orig_linear.bias is not None

        self.linear = orig_linear
        for p in self.linear.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.merge_weights = merge_weights

        # LoRA params
        if r > 0:

            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        base = self.linear(x)
        if self.r > 0:
            # x: (..., in)
            # x @ A.T -> (..., r)
            # (..., r) @ B.T -> (..., out)
            lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
            return base + lora_out * self.scaling
        else:
            return base

    def lora_state_dict(self):
        if self.r > 0:
            return {"lora_A": self.lora_A.data.cpu(), "lora_B": self.lora_B.data.cpu()}
        else:
            return {}

def apply_lora_to_module(module, target_names=("qkv", "proj", "fc", "proj"), r=8, alpha=16, inject_in_mlp=False):
    
    for name, child in module.named_children():
        #If child itself contains children, recurse
        apply_lora_to_module(child, target_names, r, alpha, inject_in_mlp)

        if isinstance(child, nn.Linear):
            lname = name.lower()
            should_wrap = False
            if any(k in lname for k in ["qkv", "proj", "attn", "to_", "toq", "tok"]):
                should_wrap = True
            if inject_in_mlp and any(k in lname for k in ["fc", "mlp", "proj"]):
                should_wrap = True

            if should_wrap:
                wrapped = LoRALinear(child, r=r, alpha=alpha)
                setattr(module, name, wrapped)

def print_trainable_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.3f}M, Trainable params: {trainable/1e6:.3f}M")
