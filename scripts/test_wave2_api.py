#!/usr/bin/env python3
"""Smoke test for Approach 1 — the Wave 2 LLM-API pipeline.

Verifies, on a small card sample against the REAL endpoint (uses .env config):
  1. connection to the gateway (GET /models) and model availability
  2. W2.1 orientation attribution runs for all 78 cards (deterministic proxy)
  3. W2.2 reversed synthesis on a 3-card sample (The Fool, Two of Cups,
     Four of Pentacles) — generation + all four gates + keyword rubric
  4. negative control (wrong-card spine must be rejected, floor >= 80%)
  5. W2.3 assembly assertions (--allow-incomplete, since only a sample is
     synthesised — the full run produces complete cards.jsonl)

Usage:
  python scripts/test_wave2_api.py [--cards 0,36,52] [--neg-control 4] [--no-cache]

Exit code 0 = all checks green. Uses the prompt-hash cache by default so
re-runs are cheap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tfvn.llm_client import LLMClient, LLMError, load_env  # noqa: E402

import build_wave2_api as b  # noqa: E402  (shares the orchestrator functions)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", default="0,36,52", help="card ids to synthesise")
    ap.add_argument("--neg-control", type=int, default=4, help="negative-control probes")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    load_env()
    fails: List[str] = []
    card_ids = [int(x) for x in args.cards.split(",") if x.strip()]

    # 1. connection ----------------------------------------------------------
    print("[1/5] connection test")
    client = LLMClient()
    if args.no_cache:
        client.cache_dir = Path("/dev/null")
    try:
        models = client.available_models()
        print(f"  OK gateway={client.base_url} models={models}")
        if client.model not in models:
            fails.append(f"model {client.model!r} not advertised: {models}")
    except LLMError as e:
        fails.append(f"connection failed: {e}")
        print("  FAIL", e)
        print("\nRESULT: FAIL")
        return 1

    # 2. W2.1 -----------------------------------------------------------------
    print("[2/5] W2.1 orientation attribution (all 78 cards)")
    inputs = b.load_inputs()
    attributions = b.run_w21(inputs, epsilon=0.15)
    n_attr = len(attributions)
    print(f"  OK {n_attr} attributions written to kb/vn_orientation_attribution.json")
    if n_attr != 78:
        fails.append(f"expected 78 attributions, got {n_attr}")

    # 3. W2.2 on sample -------------------------------------------------------
    print(f"[3/5] W2.2 reversed synthesis for cards {card_ids}")
    out22 = b.run_w22(
        client,
        inputs,
        attributions,
        card_ids,
        variants=2,
        neg_control=args.neg_control,
        dry_run=False,
        seed=42,
        temp_hi=1.0,
        temp_lo=0.7,
    )
    report = out22["report"]
    agg = report["aggregate"]
    print(f"  aggregate={agg}")
    if agg["synthetic"] != len(card_ids):
        fails.append(
            f"expected {len(card_ids)} synthetic rows, got {agg['synthetic']} "
            f"(failed={agg['failed_gate']})"
        )
        for cid in card_ids:
            pc = report["per_card"][cid]
            if pc["vi_provenance"] != "synthetic":
                fails.append(
                    f"  card {cid} ({pc['name_en']}): "
                    + "; ".join(f"v{v['variant']}:{v['status']}" for v in pc["variants"])
                )

    # Gate quality checks on accepted rows: keyword recall >= 0.7
    low_recall = []
    for cid in card_ids:
        pc = report["per_card"][cid]
        for v in pc["variants"]:
            if v.get("status") == "pass":
                rec = v["gate"]["g3"]["recall"]
                if rec < 0.7:
                    low_recall.append((pc["name_en"], v["variant"], rec))
    if low_recall:
        fails.append(f"rubric recall < 0.7 on accepted variants: {low_recall}")

    # 4. negative control ------------------------------------------------------
    print("[4/5] negative control (wrong-card spine rejection)")
    neg_rate = report["negative_control_rejection_rate"]
    print(f"  rejection rate={neg_rate} floor=0.8")
    if neg_rate is None or neg_rate < 0.8:
        fails.append(f"negative-control rejection rate {neg_rate} < 0.8 floor")

    # 5. W2.3 assembly assertions ----------------------------------------------
    print("[5/5] W2.3 assembly assertions (allow-incomplete)")
    rc = b.run_w23(inputs, attributions, out22["vn_spine"], allow_incomplete=True)
    if rc != 0:
        fails.append("W2.3 assertion suite failed (see output above)")

    print()
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("RESULT: PASS — all smoke checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
