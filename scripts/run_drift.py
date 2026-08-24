#!/usr/bin/env python3
"""Run the language-drift metric against a provider; optionally compare to a
baseline report.

  # record baseline (base model):
  python3 scripts/run_drift.py --provider hf@Qwen/Qwen3-1.7B --n 200 \
      --out evals/results/drift_baseline.json
  # compare a fine-tune:
  python3 scripts/run_drift.py --provider hf@out/ckpt --n 200 \
      --baseline evals/results/drift_baseline.json --out evals/results/drift_ft.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.drift import load_whitelist_surfaces, run as run_drift, whitelist_hash, compare  # noqa: E402
from evals.provider import provider_from_spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--whitelist", default="kb/card_name_whitelist.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--collapse-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline", default=None,
                    help="path to baseline report to compute deltas against")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    surfaces = load_whitelist_surfaces(Path(args.whitelist))
    sha = whitelist_hash(Path(args.whitelist))
    provider = provider_from_spec(args.provider)

    report = run_drift(provider, args.n, args.max_tokens, surfaces,
                       args.collapse_threshold, seed=args.seed)
    report["whitelist_sha256"] = sha

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        if base.get("whitelist_sha256") != sha:
            print("FATAL: baseline whitelist hash mismatch — fix the shared whitelist",
                  file=sys.stderr)
            return 2
        report["comparison"] = compare(base, report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    agg = report["aggregate"]
    print(f"mean_vi_frac={agg['mean_vi_frac']:.3f} collapse_rate={agg['collapse_rate']:.3f} "
          f"({args.n} gens) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
