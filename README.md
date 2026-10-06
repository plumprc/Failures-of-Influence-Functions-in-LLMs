<div align="center">

# 🔍 Do Influence Functions Work on Large Language Models?

**Official implementation of our empirical study on influence functions for LLMs**

[![arXiv](https://img.shields.io/badge/arXiv-2409.19998-b31b1b.svg)](https://arxiv.org/abs/2409.19998)

*Zhe Li · Wei Zhao · Yige Li · Jun Sun*

</div>

---

## 📌 Overview

Influence functions are a classic tool for tracing a model's prediction back to its training data.
This repository contains the code to **fine-tune LLMs with LoRA**, **compute influence scores**, and
**evaluate them against simple baselines** on three tasks:

- 🧪 **Harmful data identification**
- 🏷️ **Response class attribution**
- 🚪 **Backdoor trigger detection**

---

## 📁 Repository Structure

```
.
├── finetune.py       # 1️⃣  LoRA fine-tune a base LLM on a dataset
├── influence.py      # 2️⃣  Collect gradients & compute influence scores
├── generate.py       #     Generate responses and measure exact-match accuracy
├── repsim.py         # 📊  Representation-similarity baseline (hidden-state cosine sim)
├── delta.py          # 📉  LoRA ΔW norm statistics
├── classify.py       # 🔬  Harmfulness classification of generated responses
├── utils.py          # 🛠️  Dataset preprocessing / gradient / influence utilities
├── Lab.ipynb         # 📓  Notebook for analysis & ablation
│
├── datasets/         # 📚  Pre-built datasets (HuggingFace `datasets` format)
├── lora_adapter/     # 💾  Fine-tuned LoRA adapters   (output)
├── grad/             # 💾  Cached per-sample gradients (output)
└── cache/            # 📈  Influence scores & results
```

---

## ⚙️ Setup

```bash
git clone https://github.com/plumprc/Failures-of-Influence-Functions-in-LLMs.git
cd Failures-of-Influence-Functions-in-LLMs

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Quick Start

The pipeline is **three steps**: fine-tune → collect gradients → compute influence.

### 1️⃣ Fine-tune

```bash
python finetune.py --dataset grammars --epochs 10
# → lora_adapter/<model>/grammars_10
```

Options: `--model`, `--dataset`, `--epochs`, `--batch_size`, `--max_length`,
`--lora_r`, `--lora_alpha`, `--load_in_8bit`, `--val`, `--target_layer`.

### 2️⃣ Collect gradients

```bash
python influence.py --lora grammars_10
# → grad/<model>/grammars_10_{tr,val}.pkl
```

> This first pass only **caches gradients** and then exits — it's the expensive part,
> so you only do it once per adapter.

### 3️⃣ Compute influence scores

```bash
python influence.py --lora grammars_10 --grad_cache
```

This computes influence with each estimator and reports **accuracy / coverage**.

### 🧮 Influence-function variants

`influence.py` implements several estimators — see `utils.influence_function(..., hvp_cal=...)`:

| `hvp_cal` | Method | Notes |
|---|---|---|
| `gradient_match` | Gradient Match | first-order approximation |
| `LiSSA` | LiSSA | stochastic Hessian-vector products (`--iter`, `--alpha`) |
| `DataInf` | DataInf | closed-form; uses `--lambda_c` |
| `Original` | Exact IF | reference implementation (slow, mostly commented out) |

### 📊 Baselines

```bash
python repsim.py   --lora grammars_10   # representation similarity
python delta.py    --lora grammars_10   # LoRA ΔW norms
python generate.py --lora grammars_10   # generation accuracy
python classify.py --csv <name>         # harmful-response classification
```

---

## 🧭 Experiment Scripts

The benchmark commands for all three tasks are collected in [`scripts.md`](./scripts.md).

---

## 📝 Citation

```bibtex
@article{li2024influence,
  title   = {Do Influence Functions Work on Large Language Models?},
  author  = {Li, Zhe and Zhao, Wei and Li, Yige and Sun, Jun},
  journal = {arXiv preprint arXiv:2409.19998},
  year    = {2024}
}
```

---

## 🙏 Acknowledgements

We appreciate the following projects, which contributed valuable code and datasets:

- [DataInf](https://github.com/ykwon0407/DataInf)
- [🤗 peft](https://github.com/huggingface/peft)
- [🤗 transformers](https://github.com/huggingface/transformers)
- [🤗 datasets](https://github.com/huggingface/datasets)

---

## 📮 Contact

Questions or suggestions? Open an issue, or reach out to **zheli@smu.edu.sg**.
