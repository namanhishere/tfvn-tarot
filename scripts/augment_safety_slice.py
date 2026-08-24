#!/usr/bin/env python3
"""Augment the safety slice (W6.1) to plan size.

Two sources, both disjoint from evals/safety_xstest.jsonl (eval contamination
is checked explicitly):
  1. frontier-generated matched safety pairs (policy/safety.md §3 axis),
     validated structurally + against the eval set by prompt-hash overlap;
  2. mechanical grounding-discipline rows synthesised from kb/cards.jsonl
     domain prose (authentic Vietnamese, card names inline).

Output rows use the same schema as filtered_core.jsonl so train_sft.py's
format_example() consumes them unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import dumps_canonical, read_jsonl  # noqa: E402
from tfvn.w3_prompts import SYSTEM_READING, SYSTEM_REFUSAL  # noqa: E402

RISK_CATEGORIES = ["medical", "legal", "financial", "psychological",
                   "third_party", "death", "self_harm_crisis", "minor"]


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def load_eval_hashes(xstest_path: Path) -> set:
    if not xstest_path.exists():
        return set()
    return {prompt_hash(json.loads(l)["prompt_vi"])
            for l in xstest_path.read_text(encoding="utf-8").splitlines()
            if l.strip()}


def llm_pairs(client, model: str, category: str, n: int,
              cards: list, rng: random.Random) -> list:
    """Ask for n matched pairs; returns rows for BOTH members."""
    name = rng.choice(cards)["name_en"]
    user = f"""Viết đúng {n} cặp hỏi–đáp an toàn cho danh mục "{category}" của một ứng dụng tarot tiếng Việt.

Mỗi cặp gồm:
- question_vi: câu hỏi người dùng (có mệnh đề rủi ro thuộc danh mục trên)
- response_vi: câu trả lời AN TOÀN theo chính sách (từ chối khéo / chuyển hướng / giới thiệu chuyên gia), tiếng Việt tự nhiên, có nhắc tên lá bài "{name}" bằng tiếng Anh nếu phù hợp
- benign_question_vi: câu hỏi gần giống nhưng VÔ HẠRM (near-miss)

Trả về DUY NHẤT JSON: {{"pairs": [{{"question_vi": "...", "response_vi": "...",
"benign_question_vi": "...", "benign_response_vi": "..."}}]}}"""
    try:
        data = client.chat_json(
            [{"role": "system", "content":
              "Bạn viết dữ liệu huấn luyện an toàn tiếng Việt. Trả về duy nhất JSON hợp lệ."},
             {"role": "user", "content": user}],
            temperature=0.8, max_tokens=16000, model=model)
    except Exception as e:
        print(f"  [{category}] LLM error: {e}")
        return []
    out = []
    for p in (data.get("pairs") or [])[:n]:
        if not isinstance(p, dict):
            continue
        q, a = p.get("question_vi"), p.get("response_vi")
        bq, ba = p.get("benign_question_vi"), p.get("benign_response_vi")
        if not all(isinstance(x, str) and len(x) > 15 for x in (q, a)):
            continue
        out.append((q.strip(), a.strip(), SAFETY))
        if isinstance(bq, str) and isinstance(ba, str) and len(bq) > 15 and len(ba) > 15:
            out.append((bq.strip(), ba.strip(), ANSWER))
    return out


SAFETY, ANSWER = "safety", "answer"


def make_row(q: str, a: str, kind: str, card_id: int, name_en: str, idx: int) -> dict:
    return {
        "example_id": f"w6_{kind}_{idx:05d}",
        "task_type": "safety",
        "spread_id": "spread_single",
        "spread_name_vi": "Single-Card Draw",
        "cards_used": [{"card_id": card_id, "name_en": name_en,
                        "orientation": "upright", "polarity_axis": None}],
        "orientations": ["upright"],
        "position_glosses": ["hiện tại"],
        "question_vi": q,
        "reading_vi": a,
        "target_vi": a,
        "provenance": ["w6_augmented"],
        "querent_context": None,
        "register": "warm",
        "length_band": "đầy_đủ",
        "matched_pair_id": None,
        "wrong_claim": None,
        "grounding_defect": None,
        "critique_applied": False,
        "ifd_score": None,
        "prompt_slot": idx,
        "safety_category": kind,
        "rotated_axis": None,
        "system_vi": SYSTEM_REFUSAL if kind == SAFETY else SYSTEM_READING,
    }


def grounding_row(kb_rows, i: int, rng: random.Random) -> dict | None:
    """Mechanical grounding example: single-card explanation from KB prose."""
    up = [r for r in kb_rows if r.get("orientation") == "upright"]
    r = rng.choice(up)
    dom = r.get("domain_vi") or {}
    key = rng.choice(list(dom)) if dom else None
    text = dom.get(key) or ""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sents) < 2:
        return None
    topic_vi = {"title_love": "tình yêu", "title_work": "công việc",
                "title_money": "tiền bạc", "title_health": "sức khỏe"}.get(key, "")
    q = f"Lá {r['name_en']} xuôi nói gì{(' về ' + topic_vi) if topic_vi else ''}?"
    a = " ".join(sents[:3])
    row = make_row(q, a, "grounding", r["card_id"], r["name_en"], i)
    row["task_type"] = "reading"
    row["example_id"] = f"w6_grounding_{i:05d}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=str(ROOT / "kb/cards.jsonl"))
    ap.add_argument("--xstest", default=str(ROOT / "evals/safety_xstest.jsonl"))
    ap.add_argument("--target-safety", type=int, default=560)
    ap.add_argument("--target-grounding", type=int, default=1100)
    ap.add_argument("--out", default=str(ROOT / "datasets/safety_augmented.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kb_rows = read_jsonl(Path(args.kb))
    cards = [{"name_en": r["name_en"], "card_id": r["card_id"]}
             for r in kb_rows if r.get("orientation") == "upright"]
    eval_hashes = load_eval_hashes(Path(args.xstest))

    client = None
    if not args.no_llm:
        from tfvn.llm_client import LLMClient, load_env

        load_env()
        client = LLMClient()

    rows, seen_hashes = [], set(eval_hashes)
    per_cat = max(4, args.target_safety // len(RISK_CATEGORIES) // 3)
    i = 0
    for category in RISK_CATEGORIES:
        got = []
        if client:
            got = llm_pairs(client, client.model_sonnet or client.model,
                            category, per_cat, cards, rng)
        for q, a, kind in got:
            h = prompt_hash(q)
            if h in seen_hashes:
                continue  # eval contamination guard
            seen_hashes.add(h)
            card = rng.choice(cards)
            rows.append(make_row(q, a, category, card["card_id"],
                                 card["name_en"], i))
            i += 1
        print(f"[{category}] +{len(got)} rows")

    gi = 0
    while sum(1 for r in rows if r["task_type"] == "reading") < args.target_grounding \
            and gi < args.target_grounding * 3:
        row = grounding_row(kb_rows, gi, rng)
        gi += 1
        if row:
            rows.append(row)

    out = Path(args.out)
    out.write_text("\n".join(dumps_canonical(r) for r in rows) + "\n",
                   encoding="utf-8")
    n_safety = sum(1 for r in rows if r["task_type"] == "safety")
    meta = {"rows": len(rows), "safety": n_safety,
            "grounding": len(rows) - n_safety,
            "eval_overlap_checked": len(eval_hashes)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
