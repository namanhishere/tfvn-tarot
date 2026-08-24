"""Execute the complete training notebooks top-to-bottom with local stand-ins.

This is stronger than per-cell syntax checks: cells share one kernel, so
ordering and namespace flow are verified exactly as a human would run them.

Substitutions applied to each copy before execution:
  - the platform SETUP cell is replaced by a local equivalent that points
    CORE/BULK at this repo's datasets (no Kaggle/Drive/vast needed)
  - the training shell cell runs `train_sft.py --smoke` on Qwen3-0.6B CPU
    (~30 s) instead of the full GPU run; every other cell is untouched

The executed copies are written to tmp/ and not committed.
"""

import json
import sys
from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "training/notebooks"

LOCAL_SETUP = """\
# LOCAL TEST SETUP — same contract as the cloud setup cells.
import os, sys
os.chdir("{root}")
sys.path.insert(0, "src")
CORE = os.path.join("{root}", "datasets/filtered_core.jsonl")
BULK = os.path.join("{root}", "datasets/filtered_bulk.jsonl")
for p in (CORE, BULK):
    assert os.path.exists(p), f"missing dataset artifact: {{p}}"
print(f"local setup OK -> {{os.getcwd()}}")
"""

SMOKE_TRAIN_VARS = """\
EPOCHS = 1
DATA = CORE
MODEL = "Qwen/Qwen3-0.6B"
OUT = "artifacts/sft_r32"
print(f"SMOKE training {MODEL} on {DATA} -> {OUT}")
"""

SMOKE_TRAIN_SHELL = """\
!{sys.executable} scripts/train_sft.py --smoke \\
    --model "{MODEL}" --data "{DATA}" --out "{OUT}"
"""


def _prepare(nb_name: str, tmp_dir: Path) -> Path:
    nb = nbformat.read(NB_DIR / nb_name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if isinstance(src, list):
            src = "".join(src)
        if "/kaggle/input" in src or "drive.mount" in src or "REPO_URL" in src:
            cell.source = LOCAL_SETUP.format(root=str(ROOT))
        elif src.lstrip().startswith("%%bash"):
            # deps already present in the local venv; VERIFY_STACK asserts them
            cell.source = "# deps installed locally; skipped for execution test"
        elif "torch.cuda.is_available()" in src:
            # local variant: same import/API assertions, no bf16 gate
            cell.source = (
                "import torch, transformers, peft, trl, inspect\n"
                "from transformers import Trainer\n"
                "assert 'processing_class' in inspect.signature(Trainer.__init__).parameters\n"
                "print('STACK OK (local: GPU capability gate deferred to train_sft)')")
        elif src.lstrip().startswith("# Baseline run"):
            cell.source = SMOKE_TRAIN_VARS
        elif "--model" in src and "train_sft.py" in src and "RUN_ABLATION" not in src:
            cell.source = SMOKE_TRAIN_SHELL
        elif "RUN_ABLATION = False" in src:
            cell.source = "RUN_ABLATION = False\nABL = []\n"
        elif "/content/drive/MyDrive" in src:
            # Drive unavailable off-Colab; keep the copy step but target tmp
            cell.source = (
                "import shutil, os\n"
                "os.makedirs('/tmp/nbtest_out', exist_ok=True)\n"
                "shutil.copytree('artifacts', '/tmp/nbtest_out', "
                "dirs_exist_ok=True,\n"
                "              ignore=shutil.ignore_patterns('checkpoint-*'))\n"
                "print('adapters copied to /tmp/nbtest_out')")
    out = tmp_dir / f"exec_{nb_name}"
    nbformat.write(nb, out)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("nb_name", [
    "kaggle_train_sft.ipynb", "vast_train_sft.ipynb", "colab_train_sft.ipynb"])
def test_notebook_executes_end_to_end(nb_name, tmp_path):
    path = _prepare(nb_name, tmp_path)
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=600, kernel_name="tfvn-venv",
        resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
    errors = [o for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{nb_name}: {errors[:1]}"
    # the verification cell must have printed its success marker
    printed = "\n".join(
        o.get("text", "") for c in nb.cells if c.cell_type == "code"
        for o in c.get("outputs", []) if o.get("output_type") == "stream")
    assert "RUN VERIFIED" in printed, "post-training verification did not run green"
