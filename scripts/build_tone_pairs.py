#!/usr/bin/env python3
"""Build evals/tone_minimal_pairs.jsonl (plan: 300 items, mechanical labels).

If the authentic KB prose cannot supply enough items with >=3 real-word tone
distractors, the item count is reduced and the shortfall recorded (plan QA:
"reduce to 200 and note the sample size").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.vn_tones import build_items  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb/cards.jsonl")
    ap.add_argument("--out", default="evals/tone_minimal_pairs.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = build_items(Path(args.kb), n_items=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "requested": args.n,
            "built": len(items),
            "shortfall": max(0, args.n - len(items)),
            "seed": args.seed,
            "source": args.kb,
        }
    }
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload["_meta"], ensure_ascii=False) + "\n")
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(items)} items (requested {args.n}) -> {out}")
    if len(items) < args.n:
        print(f"NOTE: shortfall {args.n - len(items)} recorded in _meta")
    return 0 if len(items) >= min(args.n, 200) else 1


if __name__ == "__main__":
    sys.exit(main())
