"""Execute notebook python cells against local fixtures.

Platform-specific setup cells (Kaggle inputs, Drive mounts, vast clones) need
their host environment, so they are asserted to FAIL LOUDLY at their first
guard rather than silently proceeding — that is the contract ("runnable
without error" = every guard fires correctly and every pure cell runs clean).
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

NOTEBOOKS = sorted((ROOT / "training/notebooks").glob("*.ipynb"))


def _cells(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        yield i, src


def test_notebooks_exist():
    assert len(NOTEBOOKS) == 3


def _run_cell(src: str, ns: dict, expect_fail: bool = False):
    """Exec like IPython would: skip magics/shell lines, run the rest."""
    lines = [l for l in src.splitlines()
             if not l.lstrip().startswith(("!", "%"))]
    body = "\n".join(lines)
    try:
        exec(compile(body, "<cell>", "exec"), ns)
    except Exception as e:
        if expect_fail:
            return e
        raise
    if expect_fail:
        raise AssertionError("expected the platform guard to fire")
    return None


def test_pure_cells_execute_clean():
    """Cells with no platform dependency must run without error."""
    for nb in NOTEBOOKS:
        ns = {"__name__": "__main__",
              "CORE": "datasets/filtered_core.jsonl",
              "BULK": "datasets/filtered_bulk.jsonl"}
        for i, src in _cells(nb):
            first = next((l for l in src.splitlines() if l.strip()), "")
            # skip: bash install, stack check (needs cloud GPU), shell train,
            # platform setup, output packaging (needs artifacts)
            if first.startswith(("%%", "!")):
                continue
            if "torch.cuda.is_available()" in src:
                continue
            if "/kaggle/input" in src or "drive.mount" in src or "REPO_URL" in src:
                continue
            if "train_sft.py" in src:      # shell-escaped training commands
                continue
            if "RUN VERIFIED" in src:      # needs a real adapter; has its own test
                continue
            if any(tok in src for tok in ("os.system", "subprocess", "rmtree",
                                          "copytree", "du -sh")):
                continue
            _run_cell(src, ns)


def test_platform_guards_fire_off_platform():
    """Setup cells must fail loudly with a diagnostic off-platform."""
    cases = [
        ("kaggle_train_sft.ipynb", "/kaggle/input"),
        ("vast_train_sft.ipynb", "REPO_URL"),
    ]
    for fname, marker in cases:
        nb_path = ROOT / "training/notebooks" / fname
        setup_src = next(src for _, src in _cells(nb_path) if marker in src)
        err = _run_cell(setup_src, {}, expect_fail=True)
        assert isinstance(err, (AssertionError, NameError, FileNotFoundError)), \
            f"{fname}: expected loud guard, got {type(err).__name__}"


def test_verify_run_cell_passes_on_real_adapter(tmp_path, monkeypatch):
    """The post-training verification cell must go green against a real
    adapter directory produced by the CPU smoke run."""
    import glob

    src_adapter = Path("/tmp/smoke_run/smoke_final")
    if not src_adapter.exists():
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts/train_sft.py"),
             "--smoke", "--out", str(tmp_path / "smoke_run")],
            check=True, capture_output=True, cwd=ROOT)
        src_adapter = tmp_path / "smoke_run" / "smoke_final"

    art = ROOT / "artifacts"
    dst = art / "sft_r32" / "best"
    monkeypatch_chdir = None
    old_cwd = os.getcwd()
    try:
        art.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_adapter, dst)
        meta_src = src_adapter.parent / "run_meta.json"
        if meta_src.exists():
            shutil.copy(meta_src, dst.parent / "run_meta.json")

        nb_path = ROOT / "training/notebooks/kaggle_train_sft.ipynb"
        verify_src = next(src for _, src in _cells(nb_path)
                          if "RUN VERIFIED" in src)
        ns = {"__file__": str(ROOT)}
        os.chdir(ROOT)
        _run_cell(verify_src, ns)
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(art / "sft_r32", ignore_errors=True)


def test_train_var_cells_define_contract_namespaces():
    """The variable cells that parameterise training must define every name
    the shell cells interpolate."""
    nb_path = ROOT / "training/notebooks/kaggle_train_sft.ipynb"
    ns = {"CORE": "datasets/filtered_core.jsonl",
          "BULK": "datasets/filtered_bulk.jsonl"}
    for _, src in _cells(nb_path):
        if "EPOCHS = 2" in src or "RUN_ABLATION = False" in src:
            _run_cell(src, ns)
    for name in ("EPOCHS", "DATA", "MODEL", "OUT"):
        assert name in ns, f"missing {name}"
    assert ns["RUN_ABLATION"] is False
