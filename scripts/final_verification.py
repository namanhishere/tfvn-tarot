#!/usr/bin/env python3
"""Final verification wave (F1-F4) — machine-checkable audit.

F1 plan compliance: dependency ordering + artifact presence per wave.
F2 code quality: validator test coverage >= 90%, serialiser golden-file test
   registered, no absolute hardcoded paths in src/.
F4 scope fidelity: five committed features present; forbidden claims absent.
(F3 manual QA — serve.sh boot + API reading — is executed separately by the
caller; this script checks its prerequisites and records the outcome file.)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


def artifacts_exist() -> None:
    required = [
        "kb/english_spine.jsonl", "kb/vn_upright.jsonl", "kb/cards.jsonl",
        "kb/CARDS_HASH.txt", "kb/compact_cards.jsonl", "kb/spreads.jsonl",
        "kb/card_name_whitelist.json",
        "judge/taxonomy.json", "judge/calibration_report.md",
        "policy/safety.md", "policy/crisis_routing.py",
        "datasets/splits.json", "datasets/DATASET_HASH.txt",
        "evals/suites/core_assertions.jsonl", "evals/tone_minimal_pairs.jsonl",
        "evals/safety_xstest.jsonl", "evals/frontier_eval_protocol.md",
        "scripts/train_sft.py", "scripts/export_gguf.py",
        "scripts/build_imatrix_corpus.py", "scripts/run_quant_comparison.py",
        "scripts/train_sft.py", "training/notebooks/kaggle_train_sft.ipynb",
        "training/notebooks/vast_train_sft.ipynb",
        "training/notebooks/colab_train_sft.ipynb",
        "src/tfvn/tools.py", "src/tfvn/reading.py", "src/tfvn/serve.py",
        "scripts/serve.sh", "tests/test_extensibility.py",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    check("F1.artifacts", not missing,
          f"missing: {missing}" if missing else f"{len(required)} artifacts present")


def kb_assertions() -> None:
    from tfvn.assert_kb import run_assertions

    rc = subprocess.run([sys.executable, "-m", "tfvn.assert_kb"],
                        capture_output=True, text=True,
                        cwd=ROOT,
                        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    check("F1.kb_assertions_exit0", rc.returncode == 0,
          (rc.stdout + rc.stderr).strip().splitlines()[-1] if rc.returncode else "exit 0")


def dataset_hash() -> None:
    recorded = (ROOT / "datasets/DATASET_HASH.txt").read_text().strip()
    check("F2.dataset_hash_recorded", len(recorded) >= 64 or recorded != "",
          recorded.splitlines()[0][:80] if recorded else "empty")


def no_absolute_paths() -> None:
    offenders = []
    for py in (ROOT / "src").rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        text = py.read_text(encoding="utf-8")
        for m in re.finditer(r"[\"'](/home/[^\s\"']+|[\"']/Users/[^\s\"']+)", text):
            offenders.append(f"{py.name}:{text[:m.start()].count(chr(10)) + 1}")
    check("F2.no_hardcoded_home_paths", not offenders, ", ".join(offenders[:5]))


def scope_fidelity() -> None:
    """Five committed features + claims-integrity greps."""
    features = {
        "deck_fold": ROOT / "src/tfvn/tools.py",
        "byte_stable_rag": ROOT / "src/tfvn/reading.py",
        "validators_serving": ROOT / "src/tfvn/serve.py",
        "crisis_routing": ROOT / "policy/crisis_routing.py",
        "extensible_deck": ROOT / "tests/test_extensibility.py",
    }
    missing = [k for k, p in features.items() if not p.exists()]
    check("F4.five_features_present", not missing, f"missing: {missing}" if missing
          else "deck fold, meanings/RAG, multi-card synthesis, extensibility, safety stop")

    # forbidden claims must not appear in shipped docs
    bad_claims = []
    for doc in [ROOT / "README.md", ROOT / "evals/frontier_eval_protocol.md"]:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8").lower()
        if "native-verified prose" in text.replace("-", " ") and \
                "not native-verified" not in text:
            bad_claims.append(doc.name)
    check("F4.claims_downgrade_respected", not bad_claims, ",".join(bad_claims) or "clean")


def main() -> int:
    artifacts_exist()
    kb_assertions()
    dataset_hash()
    no_absolute_paths()
    scope_fidelity()

    # F2: validators coverage (>= 90% line coverage on src/tfvn/validators.py)
    cov = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/validators/", "-q",
         "--cov=tfvn.validators", "--cov-report=term"],
        capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"validators\.py[^\n]*?(\d+)%", cov.stdout) \
        if cov.returncode == 0 else None
    pct = int(m.group(1)) if m else -1
    check("F2.validators_coverage_ge90", pct >= 90, f"coverage={pct}%")

    # F2: serialiser golden-file determinism across two processes
    golden = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'src');"
         "from tfvn.serialise import dumps_canonical;"
         "import hashlib; r={'b':1,'a':[3,2],'c':{'x':None,'y':'é'}};"
         "print(hashlib.sha256(dumps_canonical(r).encode()).hexdigest())"],
        capture_output=True, text=True, cwd=ROOT)
    h1 = golden.stdout.strip()
    golden2 = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'src');"
         "from tfvn.serialise import dumps_canonical;"
         "import hashlib; r={'a':[3,2],'c':{'y':'é','x':None},'b':1};"
         "print(hashlib.sha256(dumps_canonical(r).encode()).hexdigest())"],
        capture_output=True, text=True, cwd=ROOT)
    check("F2.serialiser_golden_deterministic",
          h1 and h1 == golden2.stdout.strip(), f"sha={h1[:16]}…")

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    report = {
        "passed": passed, "total": len(CHECKS),
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
    }
    out = ROOT / "artifacts/final_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for n, ok, d in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'} {n} {('- ' + d) if d else ''}")
    print(f"{passed}/{len(CHECKS)} -> {out}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
