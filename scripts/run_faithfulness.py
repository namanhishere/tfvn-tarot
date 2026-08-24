#!/usr/bin/env python3
"""Run the faithfulness E2E gate; optionally enforce floors vs a baseline report.

  python3 scripts/run_faithfulness.py --provider hf@Qwen/Qwen3-1.7B \
      --n-per-stratum 10 --out evals/results/faithfulness_base.json
  # later, on a fine-tune:
  python3 scripts/run_faithfulness.py --provider hf@ckpt --baseline \
      evals/results/faithfulness_base.json --out evals/results/faithfulness_ft.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.faithfulness import enforce_floor, run_gate  # noqa: E402
from evals.provider import provider_from_spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--kb", default="kb/cards.jsonl")
    ap.add_argument("--n-per-stratum", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="treat this run as the baseline (skip floor enforcement)")
    ap.add_argument("--baseline-report", default=None,
                    help="path to baseline report for floor enforcement")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    provider = provider_from_spec(args.provider)
    report = run_gate(provider, Path(args.kb), n_per_stratum=args.n_per_stratum,
                      seed=args.seed, limit=args.limit)

    if args.baseline_report:
        base = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
        report["floor_check"] = enforce_floor(report, base)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"overall_pass_rate={report['overall_pass_rate']:.2%} "
          f"by_stratum={ {k: round(v['pass_rate'], 2) for k, v in report['by_stratum'].items()} }"
          f" -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
