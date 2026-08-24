#!/usr/bin/env python3
"""W4.4: select the best checkpoint from ablation-run metric reports.

Consumes one JSON per surviving run:
  {"run": "<name>", "held_out_loss": float, "judge_win_rate": float,
   "orientation_jaccard_rate": float, "tone_accuracy": float,
   "harmful_refusal": float, "benign_refusal": float}

Selection rule (plan W4.4): maximise the product constraint — held-out loss AND
frontier-judge win-rate vs base AND orientation Jaccard below threshold AND tone
above floor AND lowest benign-over-refusal gap. Tie between core-only and
core+bulk -> ship core-only (fewer artifacts, less synthetic contamination).
Writes artifacts/checkpoint_selection.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ORIENTATION_THRESHOLD = 0.24   # from title_secondary-derived gate (W2/W4.3)
TONE_FLOOR = 0.50              # absolute floor; base-model score raises it


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for p in paths:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        r["source"] = str(p)
        runs.append(r)
    return runs


def eligibility(run: dict, tone_floor: float) -> tuple[bool, list[str]]:
    reasons = []
    if run.get("orientation_jaccard_rate", 1.0) > ORIENTATION_THRESHOLD:
        reasons.append(f"orientation rate {run['orientation_jaccard_rate']:.2%} "
                       f"> {ORIENTATION_THRESHOLD:.0%}")
    if run.get("tone_accuracy", 0.0) < tone_floor:
        reasons.append(f"tone {run['tone_accuracy']:.2%} < floor")
    if run.get("held_out_loss") is None:
        reasons.append("no held-out loss recorded")
    return not reasons, reasons


def select(runs: list[dict], tone_floor: float = TONE_FLOOR) -> dict:
    scored = []
    for r in runs:
        ok, why = eligibility(r, tone_floor)
        gap = abs(r.get("benign_refusal", 0.0) - 0.10)
        # composite: judge win-rate dominates; lower loss and smaller
        # benign-refusal gap help
        composite = (
            -float(r.get("judge_win_rate") or 0.0) * 2.0
            + float(r.get("held_out_loss") or 99.0)
            + gap
        )
        src_name = Path(r.get("source") or (r["run"] + ".json")).name
        tier = ("core" if "core" in src_name and "bulk" not in src_name
                else "bulk")
        scored.append({**r, "eligible": ok, "ineligible_reasons": why,
                       "composite": round(composite, 4), "tier": tier})

    eligible = [r for r in scored if r["eligible"]]
    if not eligible:
        return {"selected": None,
                "reason": "no eligible checkpoint — all failed hard gates",
                "ranked": sorted(scored, key=lambda r: r["composite"])}

    eligible.sort(key=lambda r: r["composite"])
    best = eligible[0]
    tied = [r for r in eligible
            if abs(r["composite"] - best["composite"]) < 1e-6]
    if len(tied) > 1:
        core = next((r for r in tied if r["tier"] == "core"), None)
        if core:
            best = core
            reason = ("tie broken to core-only per plan W4.4 "
                      "(fewer artifacts, less synthetic-contamination risk)")
        else:
            reason = f"tie among {len(tied)} runs"
    else:
        reason = "best composite under all hard gates"
    return {"selected": best["run"], "chosen": best, "reason": reason,
            "ranked": sorted(scored, key=lambda r: r["composite"]),
            "pre_safety_baseline": {
                "harmful_refusal": best.get("harmful_refusal"),
                "benign_refusal": best.get("benign_refusal"),
                "note": "measured BEFORE W6 continuation; if benign >15%, "
                        "W6 targets relax per plan W6.2"}}


def write_report(result: dict, out: Path) -> None:
    lines = ["# Checkpoint selection (W4.4)", ""]
    if result["selected"] is None:
        lines += ["**No eligible checkpoint.**", "", result["reason"], ""]
    else:
        c = result["chosen"]
        lines += [
            f"**Selected:** `{result['selected']}` — {result['reason']}", "",
            "| criterion | value |",
            "|---|---|",
            f"| held-out loss | {c.get('held_out_loss')} |",
            f"| judge win-rate vs base | {c.get('judge_win_rate')} |",
            f"| orientation Jaccard violation rate | "
            f"{c.get('orientation_jaccard_rate')} (threshold "
            f"{ORIENTATION_THRESHOLD}) |",
            f"| tone accuracy | {c.get('tone_accuracy')} |",
            f"| harmful / benign refusal | {c.get('harmful_refusal')} / "
            f"{c.get('benign_refusal')} |",
            "",
            "Pre-safety baseline recorded for W6 gating.", ""]
    lines += ["## All runs", "",
              "| run | eligible | composite | notes |", "|---|---|---|---|"]
    for r in result["ranked"]:
        lines.append(f"| {r['run']} | {r['eligible']} | {r['composite']} | "
                     f"{'; '.join(r['ineligible_reasons']) or '—'} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="per-run metric JSON files")
    ap.add_argument("--tone-floor", type=float, default=TONE_FLOOR)
    ap.add_argument("--out", default=str(ROOT / "artifacts/checkpoint_selection"))
    args = ap.parse_args()

    result = select(load_runs([Path(p) for p in args.runs]), args.tone_floor)
    out = Path(args.out)
    write_report(result, out.with_suffix(".md"))
    out.with_suffix(".json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"selected={result['selected']} -> {out}.md")
    return 0 if result["selected"] else 1


if __name__ == "__main__":
    sys.exit(main())
