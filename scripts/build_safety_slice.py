#!/usr/bin/env python3
"""W6: extract the safety slice for LoRA continuation.

Plan W6.1: core safety + grounding discipline examples, ~1,500–2,500 rows,
taken from the FILTERED TRAIN split only (anchors and test/val stay untouched).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import dumps_canonical  # noqa: E402

SAFETY_TASK = "safety"
GROUNDING_QUOTA = 0.35   # grounding-discipline share inside the slice
MAX_SLICE = 2500


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(ROOT / "datasets/filtered_core.jsonl"))
    ap.add_argument("--splits", default=str(ROOT / "datasets/splits.json"))
    ap.add_argument("--out", default=str(ROOT / "datasets/safety_slice.jsonl"))
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    train_ids = {k for k, v in splits.items() if v == "train"}

    from tfvn.serialise import read_jsonl

    all_rows = read_jsonl(Path(args.core))
    train_rows = [r for r in all_rows if r["example_id"] in train_ids]

    safety = [r for r in train_rows if r.get("task_type") == SAFETY_TASK]
    grounding = [r for r in train_rows if r.get("task_type") == "reading"
                 and not r.get("grounding_defect")]
    rng = random.Random(args.seed)
    rng.shuffle(safety)
    rng.shuffle(grounding)

    n_grounding = min(len(grounding),
                      max(int((args.max - len(safety)) * GROUNDING_QUOTA
                              / (1 - GROUNDING_QUOTA)), 1))
    n_safety = min(len(safety), args.max - n_grounding)
    slice_rows = safety[:n_safety] + grounding[:n_grounding]
    rng.shuffle(slice_rows)
    slice_rows = slice_rows[:min(MAX_SLICE, len(slice_rows))]

    out = Path(args.out)
    out.write_text("\n".join(dumps_canonical(r) for r in slice_rows) + "\n",
                   encoding="utf-8")
    meta = {
        "rows": len(slice_rows),
        "safety": n_safety,
        "grounding": n_grounding,
        "from_train_only": True,
        "leak_check": {
            "in_val_or_test": sum(
                1 for r in slice_rows if splits.get(r["example_id"]) != "train"),
        },
    }
    assert meta["leak_check"]["in_val_or_test"] == 0, "slice leaked non-train rows"
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
