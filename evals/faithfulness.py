"""End-to-end faithfulness gate (plan W3e.4) — deterministic KB licensing.

For each case (a seeded draw + question sent to a provider):
  1. containment:   every drawn card is mentioned in the output
  2. no-hallucination: zero cards outside the draw appear (majors as bait +
     whitelist faux-"The X" detection via validators)
  3. positions:     every spread position is addressed
  4. orientation:   reversed draws are not described with upright keywords

Cases are stratified by spread size (1 / 3 / 10) and orientation mix.
Per-stratum pass rates are reported; the fine-tuned model's floor is the base
model's pass rate on the same strata (never regress below it).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import read_jsonl  # noqa: E402
from tfvn.validators import (  # noqa: E402
    load_whitelist,
    validate_card_name_containment,
    validate_keyword_collision,
    validate_orientation_consistency,
)

STRATA = [(1, "single"), (3, "three"), (10, "ten")]

POSITION_LABELS = {
    1: ["hiện tại"],
    3: ["quá khứ", "hiện tại", "tương lai"],
    10: ["hiện tại", "thách thức", "quá khứ", "tương lai", "trên đầu",
         "bên dưới", "lời khuyên", "môi trường", "hy vọng và nỗi sợ",
         "kết quả"],
}

QUESTION_BANK = [
    "Tình yêu của tôi sẽ ra sao?",
    "Công việc sắp tới của tôi có thuận lợi không?",
    "Tài chính của tôi trong tháng này thế nào?",
    "Sức khỏe của gia đình tôi cần lưu ý điều gì?",
]


def make_cases(kb_rows: Sequence[dict], n_per_stratum: int, seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    cases = []
    for size, stratum in STRATA:
        labels = POSITION_LABELS[size]
        for i in range(n_per_stratum):
            rows = rng.sample(kb_rows, size)
            draw = [{"card_id": r["card_id"], "name_en": r["name_en"],
                     "orientation": r["orientation"]} for r in rows]
            cases.append({
                "case_id": f"{stratum}_{i:03d}",
                "stratum": stratum,
                "size": size,
                "draw": draw,
                "positions": labels,
                "question_vi": rng.choice(QUESTION_BANK),
            })
    return cases


def format_prompt(case: dict) -> str:
    names = ", ".join(f"{d['name_en']} ({'xuôi' if d['orientation'] == 'upright' else 'đảo'})"
                      for d in case["draw"])
    pos = ", ".join(case["positions"])
    return (
        f"Tôi rút được {case['size']} lá: {names}. "
        f"Các vị trí: {pos}. Câu hỏi: {case['question_vi']} "
        "Hãy đọc bài cho từng vị trí, chỉ nói về các lá đã rút."
    )


def gate(output: str, case: dict, kb_rows: Sequence[dict], whitelist: dict) -> dict:
    failures: List[str] = []

    majors = [r["name_en"] for r in kb_rows
              if r.get("arcana") == "major" and r.get("orientation") == "upright"]
    required = [d["name_en"] for d in case["draw"]]
    forbidden_bait = [m for m in majors if m not in required]
    c = validate_card_name_containment(output, whitelist, required_names=required)
    if c["missing_required"]:
        failures.append(f"unmentioned_drawn_cards={c['missing_required']}")
    mentioned_all = set(c["mentioned"])
    hallucinated = [b for b in forbidden_bait if b in mentioned_all]
    if hallucinated or c["hallucinated_like"] or c["not_in_whitelist"]:
        failures.append(f"hallucinated={hallucinated + c['hallucinated_like'] + c['not_in_whitelist']}")

    low = output.lower()
    missing_pos = [p for p in case["positions"] if p.lower() not in low]
    if missing_pos:
        failures.append(f"missing_positions={missing_pos}")

    o = validate_orientation_consistency(output, case["draw"], kb_rows)
    if not o["ok"]:
        failures.append(f"orientation_violations={o['violations']}")

    k = validate_keyword_collision(output, case["draw"], kb_rows)
    if not k["ok"]:
        failures.append(f"keyword_collisions={k['collisions']}")

    return {"case_id": case["case_id"], "stratum": case["stratum"],
            "passed": not failures, "failures": failures}


def run_gate(provider, kb_path: Path, n_per_stratum: int = 10, seed: int = 42,
             limit: Optional[int] = None) -> dict:
    kb_rows = read_jsonl(kb_path)
    wl = load_whitelist()
    cases = make_cases(kb_rows, n_per_stratum, seed=seed)
    if limit:
        cases = cases[:limit]
    results = []
    for case in cases:
        out = provider.generate(format_prompt(case), temperature=0.7, max_tokens=900)
        results.append(gate(out, case, kb_rows, wl))

    by_stratum: Dict[str, dict] = {}
    for r in results:
        s = by_stratum.setdefault(r["stratum"], {"total": 0, "passed": 0})
        s["total"] += 1
        s["passed"] += 1 if r["passed"] else 0
    for s in by_stratum.values():
        s["pass_rate"] = s["passed"] / s["total"] if s["total"] else 0.0
    overall_passed = sum(1 for r in results if r["passed"])
    return {
        "provider": getattr(provider, "name", str(provider)),
        "seed": seed,
        "n_cases": len(results),
        "overall_pass_rate": overall_passed / len(results) if results else 0.0,
        "by_stratum": by_stratum,
        "results": results,
    }


def enforce_floor(report: dict, baseline_report: dict) -> dict:
    """Floor rule (plan): fine-tune must not regress below the base model on any
    stratum; absolute floor 0.5 where the base model scores below 50%."""
    verdicts = {}
    for stratum, stats in report["by_stratum"].items():
        base_rate = baseline_report.get("by_stratum", {}).get(stratum, {}).get("pass_rate", 0.0)
        floor = max(base_rate, 0.5)
        rate = stats["pass_rate"]
        verdicts[stratum] = {"rate": rate, "floor": floor, "meets_floor": rate >= floor}
    return {"meets_all_floors": all(v["meets_floor"] for v in verdicts.values()),
            "verdicts": verdicts}
