# Small Language Model (SLM) from Scratch with Agentic Reasoning

This repository contains an end-to-end implementation of a **Small Language Model (SLM)** built from scratch using **PyTorch**.  
The project covers **pretraining**, **fine-tuning**, **LoRA-based parameter-efficient tuning**, **custom tokenization**, and an **agentic reasoning loop** on top of the model.

The goal of this project is to understand and implement the full lifecycle of modern GPT-style models under constrained compute.

---

## Features

- Decoder-only Transformer language model
- GPT-2 BPE tokenization using `tiktoken`
- Pretraining on large-scale text corpora
- Fine-tuning on task-specific datasets
- LoRA (Low-Rank Adaptation) for efficient finetuning
- Perplexity-based evaluation
- GSM-style math evaluation
- Agentic reasoning loop with self-critique and retries
- Clean, modular, research-oriented codebase

---

## Project Structure

```text
WINTER-PROJECT-SLM/
├── model/
│   ├── model.py          # Transformer language model
│   ├── lora.py           # LoRA modules
│   └── starter.ipynb
│
├── tokenizer/
│   ├── __init__.py
│   ├── base.py
│   └── regexTokenizer.py
│
├── data/
│   ├── openwebtext.py    # Dataset processing
│   └── math.py           # Math/GSM-style data
│
├── training/
│   ├── pretraining.ipynb
│   └── finetuning.ipynb
│
├── report/
│   └── Report.pdf
│
├── README.md
└── .gitignore
```
## Requirements

The project was developed and tested with the following setup:

- Python **3.9+**
- PyTorch **2.x**
- CUDA (optional, but recommended for training)

### Python Dependencies

```text
torch
tiktoken
numpy
tqdm
matplotlib
datasets
regex
```
