"""Forgetting tripwire: Vietnamese card-knowledge MCQ, mechanically derived.

Items are built from kb/cards.jsonl `domain_vi` prose (authentic Vietnamese,
no frontier involvement): a context sentence is shown and the model must pick
which of 4 canonical names the prose describes. Distractors share arcana/suit
so surface cues don't leak the answer.

Scoring: providers with loglikelihood score each option as a continuation;
otherwise the provider generates an "A/B/C/D" letter answer which is parsed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import read_jsonl  # noqa: E402

from evals.provider import GenerationProvider  # noqa: E402

DOMAIN_KEYS = ["title_love", "title_work", "title_money", "title_health"]
_LETTER_RE = re.compile(r"\b([ABCD])\b")
_LETTER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


def build_mcq(kb_path: Path, n_items: int = 300, seed: int = 42) -> List[dict]:
    """Build n_items multiple-choice items deterministically from KB prose."""
    import random

    rows = read_jsonl(kb_path)
    upright = [r for r in rows if r.get("orientation") == "upright"]
    by_id = {int(r["card_id"]): r for r in upright}
    rng = random.Random(seed)

    pool = []
    for cid, r in sorted(by_id.items()):
        for dk in DOMAIN_KEYS:
            text = (r.get("domain_vi") or {}).get(dk) or ""
            first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
            if len(first.split()) >= 5:
                pool.append({"card_id": cid, "name_en": r["name_en"], "suit": r.get("suit"),
                             "arcana": r.get("arcana"), "domain": dk, "snippet": first})
    rng.shuffle(pool)
    pool = pool[:n_items]

    by_arcana_suit: dict = {}
    for p in sorted({q["card_id"] for q in pool}):
        q = by_id[p]
        by_arcana_suit.setdefault((q.get("arcana"), q.get("suit")), []).append(p)

    items = []
    for i, p in enumerate(pool):
        same_group = [c for c in by_arcana_suit[(p["arcana"], p["suit"])]
                      if c != p["card_id"]]
        rng.shuffle(same_group)
        distractors = same_group[:3]
        candidates = list(by_id)
        rng.shuffle(candidates)
        for c in candidates:
            if len(distractors) >= 3:
                break
            if c != p["card_id"] and c not in distractors:
                distractors.append(c)
        options = [by_id[c]["name_en"] for c in distractors]
        answer_idx = rng.randrange(4)
        options.insert(answer_idx, p["name_en"])
        items.append({
            "item_id": f"mcq_{i:04d}",
            "card_id": p["card_id"],
            "answer_name": p["name_en"],
            "answer_idx": answer_idx,
            "options": options,
            "domain": p["domain"],
            "context_vi": p["snippet"],
        })
    return items


def format_prompt(item: dict) -> str:
    opts = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(item["options"]))
    return (
        "Đoạn văn sau mô tả một lá bài tarot nào?\n\n"
        f"\"{item['context_vi']}\"\n\n"
        f"{opts}\n\n"
        "Trả lời bằng đúng một chữ cái (A, B, C hoặc D)."
    )


def score_mcq(provider: GenerationProvider, items: Sequence[dict]) -> dict:
    used_ll = bool(getattr(provider, "supports_loglikelihood", lambda: False)())
    correct = 0
    details = []
    for item in items:
        prompt = format_prompt(item)
        if used_ll:
            scores = [provider.loglikelihood(prompt + "\nĐáp án:", f" {'ABCD'[i]}")
                      for i in range(4)]
            pred = max(range(4), key=lambda i: scores[i])
        else:
            out = provider.generate(prompt, temperature=0.0, max_tokens=8)
            m = _LETTER_RE.search(out.upper())
            pred = _LETTER_MAP.get(m.group(1), -1)
        ok = pred == item["answer_idx"]
        correct += ok
        details.append({"item_id": item["item_id"], "pred": pred, "correct": ok})

    return {
        "provider": provider.name,
        "scoring": "loglikelihood" if used_ll else "letter-parse",
        "n_items": len(items),
        "accuracy": correct / len(items) if items else 0.0,
        "details": details,
    }
