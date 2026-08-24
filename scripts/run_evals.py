#!/usr/bin/env python3
"""Run an assertion suite against a provider and write a JSON report.

Usage:
  python3 scripts/run_evals.py --provider mock --suite evals/suites/core_assertions.jsonl \
      --out evals/results/core_mock.json [--limit 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.assertions import run_suite  # noqa: E402
from evals.provider import provider_from_spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, help="mock | llama-server@URL | hf@model[:dev]")
    ap.add_argument("--suite", default="evals/suites/core_assertions.jsonl")
    ap.add_argument("--kb", default="kb/cards.jsonl")
    ap.add_argument("--whitelist", default=None)
    ap.add_argument("--out", default="evals/results/assertions_report.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    provider = provider_from_spec(args.provider)
    report = run_suite(
        provider, Path(args.suite),
        kb_path=Path(args.kb),
        whitelist_path=Path(args.whitelist) if args.whitelist else None,
        limit=args.limit,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['n_passed']}/{report['n_cases']} passed "
          f"(pass_rate={report['pass_rate']:.2%}) provider={report['provider']}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
