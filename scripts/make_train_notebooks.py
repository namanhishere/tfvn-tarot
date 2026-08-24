#!/usr/bin/env python3
"""Generate Kaggle / vast.ai / Colab training notebooks (W4).

All three share the same cell plan; only the SETUP cell differs (repo access,
paths, output handling). Every code cell is syntax-checked here at generation
time, and each notebook ends with self-verification cells so a failed stage is
visible immediately.

Output: training/notebooks/{kaggle,vast,colab}_train_sft.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NB = {"cells": [], "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}, "nbformat": 4, "nbformat_minor": 5}


def md(text):
    NB["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(src):
    NB["cells"].append({"cell_type": "code", "execution_count": None,
                        "metadata": {}, "outputs": [],
                        "source": src})


def reset():
    NB["cells"] = []


# ---------------------------------------------------------------- shared ----

PINNED_INSTALL = """\
%%bash
pip install -q --upgrade pip
pip install -q torch==2.13.0 transformers==5.14.1 peft==0.20.0 trl==1.9.2 \\
    accelerate==1.14.0 datasets==5.0.1 bitsandbytes==0.50.0 safetensors==0.8.0 einops scipy
"""

VERIFY_STACK = r'''
# Verify the stack BEFORE touching weights (plan W4.1 acceptance).
import torch, transformers, peft, trl
print("torch", torch.__version__, "| tf", transformers.__version__,
      "| peft", peft.__version__, "| trl", trl.__version__)
assert torch.cuda.is_available(), "No CUDA GPU visible — select a GPU runtime."
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability()
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {name} | CC {cc} | VRAM {vram:.1f} GB")
assert cc[0] >= 8, f"CC {cc}: bf16 requires CC >= 8 (Turing/Ampere+)."
bf16_ok = torch.cuda.is_bf16_supported()
major_gate = cc[0] >= 8
assert major_gate, "Plan gate: trust get_device_capability, not is_bf16_supported()."
print("bf16:", bf16_ok, "| capability-gate:", major_gate)
# v5 API surface used by train_sft.py must exist
from transformers import Trainer  # noqa
import inspect
assert "processing_class" in inspect.signature(Trainer.__init__).parameters, \
    "transformers v5 'processing_class=' missing — wrong version installed?"
print("STACK OK")
'''

TRAIN_CELLS_MD = """\
## Train

Baseline config (plan W4.2): LoRA r=32 α=32 dropout=0 all-linear,
LR 1e-4 cosine→1%, warmup steps=3%, effective batch 32, seq 2048,
completions-only loss, epochs 2, checkpoint every ¼ epoch, seed 42.
The **epoch-1 orientation tripwire** runs inside the script: ≥5% of card pairs
exceeding Jaccard 0.24 halts the run (plan W4.3).
"""
BASELINE_TRAIN_VARS = r'''
# Baseline run (plan W4.2). Override here for ablations.
EPOCHS = 2          # stopping rule
DATA = CORE         # path to filtered_core.jsonl (set by setup cell)
MODEL = "Qwen/Qwen3-1.7B"
OUT = "artifacts/sft_r32"
print(f"training {MODEL} on {DATA} -> {OUT} for {EPOCHS} epochs")
'''

BASELINE_TRAIN_SHELL = r'''
!python scripts/train_sft.py \
    --model "{MODEL}" \
    --data "{DATA}" \
    --out "{OUT}" \
    --epochs {EPOCHS}
'''

ABLATION_TRAIN_VARS = r'''
# Optional ablation bracket (plan W4.3): rank x data scale (~6 runs).
RUN_ABLATION = False
if RUN_ABLATION:
    ABL = [(r, CORE, f"artifacts/sft_r{r}_core") for r in (16, 32, 64)] + \
          [(r, BULK, f"artifacts/sft_r{r}_bulk") for r in (16, 32, 64)]
    print("\n".join(a[2] for a in ABL))
'''

ABLATION_TRAIN_SHELL = r'''
if RUN_ABLATION:
    for r, data_path, out_dir in ABL:
        !python scripts/train_sft.py --model Qwen/Qwen3-1.7B \
            --data "{data_path}" --out "{out_dir}" \
            --epochs 2 --lora-r {r} --lora-alpha {r}
'''

VERIFY_RUN = r'''
# Post-training verification: adapter exists, metadata sane, loss recorded.
import json, glob, os
adapters = sorted(glob.glob("artifacts/*/best") + glob.glob("artifacts/*/smoke_final"))
assert adapters, "no trained adapter found under artifacts/"
latest = max(adapters, key=os.path.getmtime)
meta_path = os.path.join(os.path.dirname(latest), "run_meta.json")
meta = json.load(open(meta_path))
print("adapter :", latest)
print("loss    :", meta.get("final_loss"))
print("dtype   :", meta.get("dtype"), "| device:", meta.get("device"))
assert meta.get("final_loss") is not None, "training produced no loss record"
trip = os.path.join(os.path.dirname(latest), "orientation_tripwire_epoch1.json")
if os.path.exists(trip):
    t = json.load(open(trip))
    print(f"tripwire: rate={t['rate']:.2%} halt={t['halt']}")
    assert not t["halt"], "orientation tripwire fired — return to C2/C3"
print("RUN VERIFIED")
'''


def common_cells(repo_cell_src):
    md("# vn-tarot-llm — SFT fine-tuning\n\n"
       "Fine-tunes Qwen3-1.7B on the frozen Vietnamese tarot SFT corpus "
       "(`datasets/filtered_core.jsonl`, 11.5k rows). Implements plan waves "
       "**W4.1–W4.3**. Works identically across Kaggle / vast.ai / Colab; only "
       "the setup cell differs.")
    code(repo_cell_src)
    code(PINNED_INSTALL)
    code(VERIFY_STACK)


def tail_cells(out_cell_src):
    code(BASELINE_TRAIN_VARS)
    code(BASELINE_TRAIN_SHELL)
    code(ABLATION_TRAIN_VARS)
    code(ABLATION_TRAIN_SHELL)
    code(VERIFY_RUN)
    md("## Package outputs")
    code(out_cell_src)


# ------------------------------------------------------------------ kaggle ---

KAGGLE_SETUP = r'''
# KAGGLE SETUP — one-time preparation of the repo snapshot.
# 1. Upload this repository (without data/, .venv/, .cache/) as a PRIVATE Kaggle
#    Dataset named e.g. "<your-user>/tfvn-tarot-repo".
# 2. In the notebook's Input panel, attach that dataset.
# 3. Attach a GPU accelerator (P100 / T4 x2 / A100).
import os, shutil, sys

CANDIDATES = ["/kaggle/input/" + d for d in os.listdir("/kaggle/input")]
repo_src = next((c for c in CANDIDATES
                 if os.path.exists(os.path.join(c, "scripts/train_sft.py"))), None)
assert repo_src, ("repository snapshot not found among Kaggle inputs; "
                  f"looked in: {CANDIDATES}")

WORK = "/kaggle/working/tfvn-tarot"
if os.path.exists(WORK):
    shutil.rmtree(WORK)
shutil.copytree(repo_src, WORK, ignore=shutil.ignore_patterns(
    ".git", ".venv", ".cache", "__pycache__", "data"))
os.chdir(WORK)
sys.path.insert(0, "src")

CORE = os.path.join(WORK, "datasets/filtered_core.jsonl")
BULK = os.path.join(WORK, "datasets/filtered_bulk.jsonl")
for p in (CORE, BULK):
    assert os.path.exists(p), f"missing dataset artifact: {p}"
n_rows = sum(1 for _ in open(CORE))
print(f"repo -> {WORK} | core rows: {n_rows}")
'''

KAGGLE_OUT = r'''
import shutil
# /kaggle/working persists as the run's output — keep it small (adapters only).
shutil.rmtree("/kaggle/working/artifacts/sft_r32/checkpoint-*", ignore_errors=True)
print("output size:")
os.system("du -sh /kaggle/working/artifacts/* 2>/dev/null")
'''


# -------------------------------------------------------------------- vast ---

VAST_SETUP = r'''
# VAST.AI SETUP — assumes a PyTorch/Jupyter template instance with internet.
# Run this notebook from /workspace (the default working directory).
import os, subprocess, sys

if not os.path.exists("tfvn-tarot/scripts/train_sft.py"):
    # paste YOUR repo URL here (GitHub/GitLab); the repo must include
    # datasets/filtered_core.jsonl + filtered_bulk.jsonl
    REPO_URL = ""  # e.g. "https://github.com/you/tfvn-tarot.git"
    assert REPO_URL, "set REPO_URL above (or pre-clone the repo into ./tfvn-tarot)"
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "tfvn-tarot"],
                   check=True)
os.chdir("tfvn-tarot")
sys.path.insert(0, "src")

CORE = os.path.abspath("datasets/filtered_core.jsonl")
BULK = os.path.abspath("datasets/filtered_bulk.jsonl")
for p in (CORE, BULK):
    assert os.path.exists(p), f"missing dataset artifact: {p}"
print(f"repo -> {os.getcwd()} | free disk below.")
os.system("df -h /workspace | tail -1")
'''

VAST_OUT = r'''
import subprocess
# adapters live under /workspace/tfvn-tarot/artifacts — rsync off-box or keep.
subprocess.run(["du", "-sh", "artifacts/*"], cwd=os.getcwd())
print("remember: vast instances are ephemeral — download adapters when done.")
'''


# ------------------------------------------------------------------- colab ---

COLAB_SETUP = r'''
# COLAB SETUP — repo + datasets live in Google Drive.
# Expected Drive layout:  /content/drive/MyDrive/tfvn-tarot/{scripts,src,datasets,kb,...}
from google.colab import drive  # type: ignore
import os, sys

drive.mount("/content/drive")
REPO = "/content/drive/MyDrive/tfvn-tarot"
assert os.path.exists(os.path.join(REPO, "scripts/train_sft.py")), \
    f"repo not found at {REPO} — upload it (without data/, .venv/, .cache/)"
os.chdir(REPO)
sys.path.insert(0, "src")

CORE = os.path.join(REPO, "datasets/filtered_core.jsonl")
BULK = os.path.join(REPO, "datasets/filtered_bulk.jsonl")
for p in (CORE, BULK):
    assert os.path.exists(p), f"missing dataset artifact: {p}"
print(f"repo -> {REPO}")
os.system("nvidia-smi -L")
'''

COLAB_OUT = r'''
# HF cache lives on the ephemeral disk; adapters go back to Drive.
import shutil, os
dst = "/content/drive/MyDrive/tfvn-tarot/artifacts"
shutil.copytree("artifacts", dst, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("checkpoint-*"))
print(f"adapters copied to {dst}")
'''


def make(path: Path, setup: str, out_cell: str, extra_md: str):
    reset()
    md(extra_md)
    common_cells(setup)
    tail_cells(out_cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(NB, indent=1), encoding="utf-8")
    print("wrote", path)


def main() -> int:
    out = ROOT / "training/notebooks"
    make(out / "kaggle_train_sft.ipynb", KAGGLE_SETUP, KAGGLE_OUT,
         "*Runs on Kaggle: attach the repo snapshot as an input Dataset and a "
         "GPU accelerator.*\n")
    make(out / "vast_train_sft.ipynb", VAST_SETUP, VAST_OUT,
         "*Runs on vast.ai: any PyTorch/Jupyter template with ≥24 GB VRAM "
         "(RTX 3090/4090/A5000…). Set `REPO_URL` in the setup cell.*\n")
    make(out / "colab_train_sft.ipynb", COLAB_SETUP, COLAB_OUT,
         "*Runs on Google Colab (Pro recommended for A100/L4): repo lives in "
         "Google Drive at `MyDrive/tfvn-tarot`.*\n")
    for p in sorted(out.glob("*.ipynb")):
        nb = json.loads(p.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = cell["source"]
            if isinstance(src, list):
                src = "".join(src)
            lines = src.lstrip().splitlines() if src.strip() else []
            if not lines:
                continue
            # IPython shell escapes (`!cmd`, `%magic`) are valid in notebooks
            # but not plain python — skip syntax-checking those cells.
            if lines[0].startswith(("%%", "!", "%")) or \
               any(l.lstrip().startswith("!") for l in lines):
                continue
            compile(src, f"{p.name}#{i}", "exec")
        print(f"VALIDATED {p.name}: {len(nb['cells'])} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
