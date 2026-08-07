#!/usr/bin/env python3
"""Wave 3 orchestrator — SFT dataset generation (W3.2) + anti-collapse (W3.3).

Enumerates the prompt cross-product (plan W3.2):
  card x orientation x position semantics x spread type x querent context
  (love/career/money/health/spiritual/decision) x register (formal/warm/casual)
  x length band (ngắn/đầy_đủ) x interaction type
  (explanation/reading/followup/refusal/correction).

Two-stage generation (Magpie split): question draw at HIGH temperature (1.0),
reading draw at LOW temperature (0.7), in SEPARATE calls. Safety slice uses
MATCHED pairs (identical card/spread/position/topic, differing only in the
risk-bearing clause). Grounding-discipline negatives, counter-sycophancy, and
offline frontier critique-and-revise (critique on SONNET, revise on HAIKU —
the approved W3 mix) are included. ALL generations cached by prompt hash and
filtered through the W3.3 anti-collapse stack (memory index + rotating prompts
+ self-tightening n-gram blacklist).

Usage:
  python scripts/build_wave3.py w32 --limit 50          # generate N examples
  python scripts/build_wave3.py w32 --resume            # continue appending
  python scripts/build_wave3.py w33                     # anti-collapse ablation
  python scripts/build_wave3.py w34                     # 4-layer filter
  python scripts/build_wave3.py w35                     # dedup + coverage
  python scripts/build_wave3.py w36                     # stratified splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.llm_client import LLMClient, LLMError, load_env  # noqa: E402
from tfvn.serialise import dumps_canonical, read_jsonl, write_jsonl  # noqa: E402
from tfvn.aliases import CANONICAL_NAMES, NAME_TO_ID  # noqa: E402
from tfvn.w3_anticollapse import (  # noqa: E402
    CharNGramEmbedding,
    MemoryIndex,
    RotatingPromptState,
    SelfTighteningNGramBlacklist,
    distinct_n,
    run_ablation,
)
from tfvn.w3_prompts import (  # noqa: E402
    build_correction_reading_messages,
    build_critique_messages,
    build_grounding_reading_messages,
    build_question_draw_messages,
    build_reading_draw_messages,
    build_refusal_messages,
    build_revise_messages,
    build_rubric_messages,
    build_safety_reading_messages,
)

KB = ROOT / "kb"
DATA_DIR = ROOT / "datasets"
RAW_DIR = DATA_DIR / "raw"
CACHE = ROOT / ".cache" / "w3"

# Safety policy categories (policy/safety.md v0.5.0) — template + behavior.
SAFETY_CATEGORIES: Dict[str, Dict[str, str]] = {
    "medical": {
        "template": (
            "Tôi không phải bác sĩ và không thể chẩn đoán bệnh. Lá bài chỉ phản ánh "
            "khía cạnh tinh thần và cảm xúc. Nếu bạn có triệu chứng sức khỏe, hãy đến "
            "cơ sở y tế, hoặc gọi 115 trong trường hợp cấp cứu."
        ),
        "benign_clause": "Tôi đang lo lắng về sức khỏe của mình dạo này.",
        "risk_clause": "Lá bài có nói tôi bị bệnh gì không? Tôi có nên đi khám ung thư không?",
    },
    "legal": {
        "template": (
            "Tôi không phải luật sư và không thể đưa ra lời khuyên pháp lý. Lá bài nói "
            "về hướng tinh thần của tình huống. Về vấn đề pháp lý, bạn nên hỏi ý kiến "
            "luật sư hoặc cơ quan có thẩm quyền."
        ),
        "benign_clause": "Tôi đang phân vân về một quyết định liên quan đến công việc.",
        "risk_clause": "Tôi có thắng kiện không? Lá bài có cho tôi khởi kiện không?",
    },
    "financial": {
        "template": (
            "Tôi không phải cố vấn tài chính. Lá bài phản ánh tâm lý và hướng đi, không "
            "phải dự đoán lời lỗ. Trước quyết định đầu tư lớn, hãy tham khảo chuyên gia tài chính."
        ),
        "benign_clause": "Tôi đang cân nhắc một khoản chi tiêu lớn sắp tới.",
        "risk_clause": "Lá bài có đảm bảo tôi đầu tư chắc thắng, sinh lời không?",
    },
    "psychological": {
        "template": (
            "Bạn đang trải qua những cảm xúc rất thật. Lá bài gợi ý cách nhìn để bạn "
            "vững vàng hơn, chứ không phải chẩn đoán tâm lý. Nếu cần người đồng hành, "
            "hãy tìm đến chuyên gia tâm lý hoặc người thân đáng tin cậy."
        ),
        "benign_clause": "Dạo này tôi thấy tinh thần có phần chùng xuống.",
        "risk_clause": "Tôi có bị trầm cảm không? Lá bài có chẩn đoán bệnh tâm lý của tôi không?",
    },
    "third_party": {
        "template": (
            "Tôi không thể nói thay suy nghĩ hoặc số phận của người khác. Lá bài cho "
            "bạn nhìn rõ chính mình trong mối quan hệ này. Hãy tập trung vào cảm xúc và "
            "lựa chọn của bạn."
        ),
        "benign_clause": "Tôi muốn hiểu rõ hơn về mối quan hệ hiện tại của mình.",
        "risk_clause": "Anh ấy/cô ấy có còn yêu tôi không? Người đó đang nghĩ gì về tôi?",
    },
    "death": {
        "template": (
            "Lá bài Death không báo hiệu cái chết thể xác. Nó nói về sự kết thúc một "
            "giai đoạn và sự chuyển hóa để tái sinh điều mới."
        ),
        "benign_clause": "Tôi sắp kết thúc một giai đoạn trong đời và thấy bồi hồi.",
        "risk_clause": "Lá Death có báo tôi hoặc người thân sắp chết không?",
    },
    "self_harm_crisis": {
        "template": (
            "Bạn đang gặp khủng hoảng, và điều đó rất quan trọng. Nếu đây là tình huống "
            "khẩn cấp, hãy gọi ngay 115 (cấp cứu y tế, 24/7). Đường dây nóng Ngày mai "
            "(096 306 1414) mở cửa 13:00–20:30 từ Thứ Tư đến Chủ Nhật. Bạn không đơn độc."
        ),
        "benign_clause": "Tôi đang trải qua một giai đoạn khó khăn nhưng vẫn ổn.",
        "risk_clause": "Tôi cảm thấy muốn tự làm hại bản thân, tôi phải làm sao?",
    },
    "minor": {
        "template": (
            "Bài trải bài này phù hợp với người từ 18 tuổi trở lên. Nếu bạn dưới 18 "
            "tuổi, hãy nhờ bố mẹ hoặc người giám hộ cùng tham gia."
        ),
        "benign_clause": "",
        "risk_clause": "Tôi mới 15 tuổi nhưng tôi muốn xem bài được không?",
    },
    "benign_dark_card": {
        "template": (
            "Lá bài này trong bối cảnh của bạn nói về sự chuyển hóa. Nó mở ra một hướng "
            "mới thay vì điềm xấu."
        ),
        "benign_clause": "Tôi rút được lá Death nhưng tôi không sợ, chỉ tò mò.",
        "risk_clause": "Tôi rút được lá Death, tôi sợ đó là điềm báo xấu cho gia đình.",
    },
}

# Grounding-discipline negative defects (plan W3.2).
GROUNDING_DEFECTS = [
    {"kind": "missing_card"},
    {"kind": "incomplete_context"},
    {"kind": "empty_gloss"},
]

# Counter-sycophancy wrong claims keyed by (card_id, orientation).
WRONG_CLAIMS = [
    ("The Fool", "upright", "Nghe nói The Fool nghĩa là tôi sắp gặp xui xẻo lớn phải không?"),
    ("Death", "upright", "Lá Death chắc chắn là báo hiệu có người thân qua đời đúng không?"),
    ("The Tower", "upright", "The Tower nghĩa là mọi thứ của tôi sắp sụp đổ hoàn toàn phải không?"),
    ("The Devil", "reversed", "The Devil ngược nghĩa là tôi đang bị nguyền rủa phải không?"),
]

CONTEXTS = [
    ("love", "tình yêu và các mối quan hệ"),
    ("career", "sự nghiệp và công việc"),
    ("money", "tiền bạc và tài chính"),
    ("health", "sức khỏe và tinh thần"),
    ("spiritual", "phát triển tâm linh"),
    ("decision", "quyết định và hướng đi"),
]
REGISTERS = ["formal", "warm", "casual"]
LENGTH_BANDS = ["ngắn", "đầy_đủ"]
INTERACTIONS = ["explanation", "reading", "followup", "refusal", "correction"]


# ------------------------------------------------------------------ loading --


def load_cards() -> List[Dict[str, Any]]:
    return read_jsonl(KB / "cards.jsonl")


def load_spreads() -> List[Dict[str, Any]]:
    return read_jsonl(KB / "spreads.jsonl")


def load_dendory_profile() -> Dict[str, Any]:
    return json.loads((KB / "dendory_structural_profile.json").read_text(encoding="utf-8"))


def load_exemplars(n: int = 4, rng: Optional[random.Random] = None) -> List[str]:
    """Authentic Vietnamese register exemplars (title_secondary) for the style."""
    rng = rng or random.Random(42)
    rows = read_jsonl(KB / "vn_upright.jsonl")
    pool = [
        (r.get("title_secondary") or "").strip()
        for r in rows
        if (r.get("title_secondary") or "").strip()
    ]
    rng.shuffle(pool)
    return pool[:n]


def card_lookup(cards: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    return {(r["card_id"], r["orientation"]): r for r in cards}


# ------------------------------------------------------------- draw planning --


def plan_draw(
    rng: random.Random,
    spreads: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
    card_weights: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[str, Any], List[int], List[str]]:
    """Pick a spread + draw cards + orientation assignment.

    Spread choice is weighted toward single/three (Dendory dominant), with a
    long tail over the larger spreads for coverage. Card ids are sampled from
    the Dendory per-card weights (fallback uniform); orientations are biased to
    ~50% reversed so orientation-salience is well represented (plan: reweight
    toward orientation-salient questions).
    """
    names = list(spreads)
    rng.shuffle(names)
    spread = names[0]
    for s in spreads:
        if s["spread_id"] in ("spread_single", "spread_three") and rng.random() < 0.75:
            spread = s
            break
    n = int(spread["cards_drawn"])
    ids = list(range(78))
    if n <= len(ids):
        rng.shuffle(ids)
        chosen = ids[:n]
        if card_weights:
            chosen = rng.choices(
                ids, weights=[card_weights.get(i, 1.0) for i in ids], k=n
            )
            if len(set(chosen)) < n:
                chosen = ids[:n]  # fall back to uniform no-replacement
    else:
        chosen = [rng.choice(ids) for _ in range(n)]
    orient = ["upright", "reversed"]
    orientations = [orient[0] if rng.random() < 0.5 else orient[1] for _ in range(n)]
    return spread, chosen, orientations


# ---------------------------------------------------------------- generation --


def _chat_json(client: LLMClient, messages: List[Dict[str, str]], *,
               max_tokens: int, temperature: float, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for budget in (max_tokens, max_tokens * 2):
        try:
            return client.chat_json(
                messages, max_tokens=budget, temperature=temperature, model=model
            )
        except LLMError:
            continue
    return None


def generate_example(
    client: LLMClient,
    rng: random.Random,
    cards: Sequence[Dict[str, Any]],
    lookup: Dict[Tuple[int, str], Dict[str, Any]],
    spreads: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
    exemplars: Sequence[str],
    slot: int,
    card_weights: Optional[Dict[int, float]] = None,
    *,
    critique_fraction: float,
    model_sonnet: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Generate ONE example across the cross-product (two-stage + optional C&R).

    Returns a row dict or None on hard failure. All LLM calls are cached by
    prompt hash; ``slot`` rotates the prompt so re-draws miss the cache.
    """
    spread, ids, orientations = plan_draw(rng, spreads, profile, card_weights)
    ctx_id, ctx_vi = rng.choice(CONTEXTS)
    register = rng.choice(REGISTERS)
    length_band = rng.choice(LENGTH_BANDS)
    interaction = rng.choice(INTERACTIONS[:2])  # explanation/reading core

    rows = [
        lookup.get((cid, orient)) for cid, orient in zip(ids, orientations)
    ]
    if any(r is None for r in rows):
        return None
    rows = [r for r in rows if r is not None]  # type: ignore[assignment]

    glosses = [p.get("label_vi") or p.get("gloss_compact") or p.get("label_en") or ""
               for p in spread["positions"]]
    if len(glosses) < len(rows):
        glosses = (glosses + [glosses[-1]] * len(rows))[: len(rows)]

    # Stage 1: question draw (HIGH temperature — Magpie split).
    q_msg = build_question_draw_messages(
        ctx_id, register, length_band, interaction,
        profile.get("vi_question_seeds") or [], slot, ctx_vi,
    )
    q_data = _chat_json(client, q_msg, max_tokens=300, temperature=1.0)
    if q_data is None:
        return None
    question_vi = str(q_data.get("question_vi") or "").strip()
    if not question_vi:
        return None

    # Stage 2: reading draw (LOW temperature).
    r_msg = build_reading_draw_messages(
        question_vi, rows, spread.get("name_vi") or spread["name_en"], glosses,
        register, length_band, slot, exemplars,
    )
    r_data = _chat_json(client, r_msg, max_tokens=700, temperature=0.7)
    if r_data is None:
        return None
    reading_vi = str(r_data.get("reading_vi") or "").strip()
    if not reading_vi:
        return None

    example: Dict[str, Any] = {
        "example_id": f"w32_{slot:06d}",
        "task_type": interaction,
        "card_ids": ids,
        "orientations": orientations,
        "spread_id": spread["spread_id"],
        "spread_name_vi": spread.get("name_vi") or spread["name_en"],
        "position_glosses": glosses,
        "querent_context": ctx_id,
        "register": register,
        "length_band": length_band,
        "question_vi": question_vi,
        "reading_vi": reading_vi,
        "target_vi": reading_vi,
        "cards_used": [
            {"card_id": r["card_id"], "name_en": r["name_en"],
             "orientation": r["orientation"], "polarity_axis": r.get("polarity_axis")}
            for r in rows
        ],
        "critique_applied": False,
        "safety_category": None,
        "matched_pair_id": None,
        "grounding_defect": None,
        "wrong_claim": None,
        "provenance": ["w32_generated"],
        "prompt_slot": slot,
    }

    # Critique-and-revise: critique on SONNET, revise on HAIKU (approved mix).
    if rng.random() < critique_fraction:
        crit_data = _chat_json(
            client, build_critique_messages(question_vi, rows, reading_vi),
            max_tokens=400, temperature=0.0, model=model_sonnet,
        )
        if crit_data is not None:
            verdict = str(crit_data.get("verdict") or "fix")
            example["critique"] = {
                "verdict": verdict,
                "issues": crit_data.get("issues") or [],
                "orientation_ok": bool(crit_data.get("orientation_ok", False)),
                "faithful": bool(crit_data.get("faithful", False)),
                "answers_question": bool(crit_data.get("answers_question", False)),
                "vietnamese_natural": bool(crit_data.get("vietnamese_natural", False)),
            }
            if verdict == "fix":
                rev_data = _chat_json(
                    client, build_revise_messages(question_vi, rows, reading_vi, crit_data, slot),
                    max_tokens=700, temperature=0.5,
                )
                if rev_data is not None:
                    revised = str(rev_data.get("reading_vi") or "").strip()
                    if revised:
                        example["reading_vi_original"] = reading_vi
                        example["reading_vi"] = revised
                        example["target_vi"] = revised
                        example["critique_applied"] = True
    return example


# -------------------------------------------------------------- safety slice --


def generate_safety_examples(
    client: LLMClient,
    rng: random.Random,
    cards: Sequence[Dict[str, Any]],
    lookup: Dict[Tuple[int, str], Dict[str, Any]],
    spreads: Sequence[Dict[str, Any]],
    slot_start: int,
    *,
    model_sonnet: Optional[str],
    per_category: int = 2,
) -> List[Dict[str, Any]]:
    """MATCHED safety pairs: same card/spread/position/topic, differing only in
    the risk-bearing clause (plan W3.2 + policy/safety.md §3)."""
    out: List[Dict[str, Any]] = []
    slot = slot_start
    for cat, spec in SAFETY_CATEGORIES.items():
        for pair_i in range(per_category):
            cid = rng.choice(range(78))
            orient = "upright" if rng.random() < 0.5 else "reversed"
            row = lookup.get((cid, orient))
            if row is None:
                continue
            spread = next((s for s in spreads if s["spread_id"] == "spread_single"), spreads[0])
            glosses = [p.get("label_vi") or p.get("label_en") or "" for p in spread["positions"]]
            for member, clause in (("benign", spec["benign_clause"]), ("risk", spec["risk_clause"])):
                if not clause:
                    continue
                question_vi = clause
                msg = build_safety_reading_messages(
                    question_vi, [row], spread.get("name_vi") or spread["name_en"],
                    glosses, cat, spec["template"], slot,
                )
                data = _chat_json(client, msg, max_tokens=700, temperature=0.7)
                if data is None:
                    continue
                reading_vi = str(data.get("reading_vi") or "").strip()
                if not reading_vi:
                    continue
                pair_id = f"safe_{cat}_{pair_i}"
                out.append({
                    "example_id": f"w32_{slot:06d}",
                    "task_type": "safety",
                    "card_ids": [cid],
                    "orientations": [orient],
                    "spread_id": spread["spread_id"],
                    "spread_name_vi": spread.get("name_vi") or spread["name_en"],
                    "position_glosses": glosses,
                    "querent_context": cat,
                    "register": "warm",
                    "length_band": "ngắn",
                    "question_vi": question_vi,
                    "reading_vi": reading_vi,
                    "target_vi": reading_vi,
                    "cards_used": [
                        {"card_id": row["card_id"], "name_en": row["name_en"],
                         "orientation": row["orientation"], "polarity_axis": row.get("polarity_axis")}
                    ],
                    "critique_applied": False,
                    "safety_category": cat,
                    "matched_pair_id": pair_id,
                    "matched_member": member,
                    "grounding_defect": None,
                    "wrong_claim": None,
                    "provenance": ["w32_generated", "safety_slice"],
                    "prompt_slot": slot,
                })
                slot += 1
    return out


def generate_grounding_examples(
    client: LLMClient,
    rng: random.Random,
    cards: Sequence[Dict[str, Any]],
    lookup: Dict[Tuple[int, str], Dict[str, Any]],
    spreads: Sequence[Dict[str, Any]],
    exemplars: Sequence[str],
    slot_start: int,
    n: int = 6,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    slot = slot_start
    for i in range(n):
        defect = rng.choice(GROUNDING_DEFECTS)
        cid = rng.choice(range(78))
        row = lookup.get((cid, "upright"))
        if row is None:
            continue
        spread = next((s for s in spreads if s["spread_id"] == "spread_single"), spreads[0])
        glosses = [p.get("label_vi") or p.get("label_en") or "" for p in spread["positions"]]
        q = rng.choice([
            "Lá bài này nói gì về tình hình của tôi?",
            "Bạn giải thích giúp tôi lá này nghĩa là gì?",
        ])
        rows = [row]
        if defect["kind"] == "empty_gloss":
            glosses = [""]
        msg = build_grounding_reading_messages(
            q, rows, spread.get("name_vi") or spread["name_en"], glosses, defect, slot,
        )
        data = _chat_json(client, msg, max_tokens=700, temperature=0.7)
        if data is None:
            continue
        reading_vi = str(data.get("reading_vi") or "").strip()
        if not reading_vi:
            continue
        out.append({
            "example_id": f"w32_{slot:06d}",
            "task_type": "grounding",
            "card_ids": [cid],
            "orientations": ["upright"],
            "spread_id": spread["spread_id"],
            "spread_name_vi": spread.get("name_vi") or spread["name_en"],
            "position_glosses": glosses,
            "querent_context": "decision",
            "register": "warm",
            "length_band": "ngắn",
            "question_vi": q,
            "reading_vi": reading_vi,
            "target_vi": reading_vi,
            "cards_used": [
                {"card_id": row["card_id"], "name_en": row["name_en"],
                 "orientation": row["orientation"], "polarity_axis": row.get("polarity_axis")}
            ],
            "critique_applied": False,
            "safety_category": None,
            "matched_pair_id": None,
            "grounding_defect": defect["kind"],
            "wrong_claim": None,
            "provenance": ["w32_generated", "grounding_negative"],
            "prompt_slot": slot,
        })
        slot += 1
    return out


def generate_correction_examples(
    client: LLMClient,
    rng: random.Random,
    cards: Sequence[Dict[str, Any]],
    lookup: Dict[Tuple[int, str], Dict[str, Any]],
    spreads: Sequence[Dict[str, Any]],
    exemplars: Sequence[str],
    slot_start: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    slot = slot_start
    for name_en, orient, wrong in WRONG_CLAIMS:
        row = next((r for r in cards if r["name_en"] == name_en and r["orientation"] == orient), None)
        if row is None:
            continue
        spread = next((s for s in spreads if s["spread_id"] == "spread_single"), spreads[0])
        glosses = [p.get("label_vi") or p.get("label_en") or "" for p in spread["positions"]]
        msg = build_correction_reading_messages(
            wrong, wrong, [row], spread.get("name_vi") or spread["name_en"], glosses, slot,
        )
        data = _chat_json(client, msg, max_tokens=700, temperature=0.7)
        if data is None:
            continue
        reading_vi = str(data.get("reading_vi") or "").strip()
        if not reading_vi:
            continue
        out.append({
            "example_id": f"w32_{slot:06d}",
            "task_type": "correction",
            "card_ids": [row["card_id"]],
            "orientations": [orient],
            "spread_id": spread["spread_id"],
            "spread_name_vi": spread.get("name_vi") or spread["name_en"],
            "position_glosses": glosses,
            "querent_context": "decision",
            "register": "warm",
            "length_band": "ngắn",
            "question_vi": wrong,
            "reading_vi": reading_vi,
            "target_vi": reading_vi,
            "cards_used": [
                {"card_id": row["card_id"], "name_en": row["name_en"],
                 "orientation": row["orientation"], "polarity_axis": row.get("polarity_axis")}
            ],
            "critique_applied": False,
            "safety_category": None,
            "matched_pair_id": None,
            "grounding_defect": None,
            "wrong_claim": wrong,
            "provenance": ["w32_generated", "counter_sycophancy"],
            "prompt_slot": slot,
        })
        slot += 1
    return out


# ---------------------------------------------------------------------- w32 --


def run_w32(args: argparse.Namespace) -> int:
    print("== W3.2 generation (two-stage, anti-collapse, safety/correction/grounding)")
    load_env()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    client = LLMClient(seed=args.seed)
    model_sonnet = client.model_sonnet or None
    if model_sonnet:
        print(f"  mix: generation={client.model} critique/judge={model_sonnet}")
    else:
        print(f"  WARNING: LLM_MODEL_SONNET unset — single-tier (generation={client.model})")

    cards = load_cards()
    lookup = card_lookup(cards)
    spreads = load_spreads()
    profile = load_dendory_profile()
    card_weights = {i: profile.get("card_sampling_weights", {}).get(name, 1.0)
                    for i, name in enumerate(
                        json.loads((KB / "card_name_whitelist.json").read_text("utf-8"))["canonical_names"]
                    )}

    out_path = RAW_DIR / "generated.jsonl"
    existing = read_jsonl(out_path) if out_path.exists() else []
    seen_ids = {e["example_id"] for e in existing}
    slot = max((int(e["prompt_slot"]) for e in existing), default=0) + 1

    memory = MemoryIndex(CACHE / "memory.jsonl")
    blacklist = SelfTighteningNGramBlacklist(
        profile_path=KB / "vn_register_profile.json"
    )
    blacklist.update([e["target_vi"] for e in existing])
    rotator = RotatingPromptState(memory, blacklist)

    exemplars = load_exemplars(4, random.Random(args.seed + slot))

    rows: List[Dict[str, Any]] = list(existing)
    limit = args.limit if args.limit else 40
    target = len(existing) + limit
    print(f"  existing={len(existing)} target={target} (limit={limit}) workers={args.workers}")

    safety_n = args.safety_pairs if args.safety_pairs is not None else 2
    grounding_n = args.grounding if args.grounding is not None else 6

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    def work_one(slot_i: int, axis: Dict[str, str]) -> Optional[Dict[str, Any]]:
        rng_i = random.Random(args.seed * 7919 + slot_i)
        ex = generate_example(
            client, rng_i, cards, lookup, spreads, profile, exemplars, slot_i,
            card_weights, critique_fraction=args.critique_fraction,
            model_sonnet=model_sonnet,
        )
        if ex is None:
            return None
        ex["rotated_axis"] = axis
        return ex

    pending = max(0, target - len(rows))
    slots = list(range(slot + 1, slot + 1 + pending * 2))  # over-provision for rejections
    axes = [rotator.next_axis() for _ in slots]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work_one, s, a): s for s, a in zip(slots, axes)}
        for fut in as_completed(futs):
            ex = fut.result()
            if ex is None:
                continue
            with lock:
                target_vi = ex["target_vi"]
                if memory.is_dup(target_vi):
                    continue
                memory.add(target_vi, {"example_id": ex["example_id"], "task_type": ex["task_type"]})
                blacklist.update([target_vi])
                rows.append(ex)
                seen_ids.add(ex["example_id"])
            if len(rows) % 10 == 0 or len(rows) >= target:
                with lock:
                    write_jsonl(out_path, rows)
                print(f"  ... {len(rows)}/{target} accepted")

    slot = max((int(e["prompt_slot"]) for e in rows), default=slot)
    write_jsonl(out_path, rows)

    # Special slices. Idempotent on resume: skip a slice family whose provenance
    # tag is already present (matched pairs must never duplicate across runs).
    base_slot = slot
    new_rows: List[Dict[str, Any]] = []
    has_safety = any("safety_slice" in (r.get("provenance") or []) for r in rows)
    has_grounding = any("grounding_negative" in (r.get("provenance") or []) for r in rows)
    has_correction = any("counter_sycophancy" in (r.get("provenance") or []) for r in rows)
    if not has_safety:
        safety_rows = generate_safety_examples(
            client, random.Random(args.seed + base_slot), cards, lookup, spreads, base_slot + 1,
            model_sonnet=model_sonnet, per_category=safety_n,
        )
        base_slot += len(safety_rows)
        new_rows.extend(safety_rows)
    else:
        print("  safety slice already present — skipping (resume)")
    if not has_grounding:
        grounding_rows = generate_grounding_examples(
            client, random.Random(args.seed + base_slot), cards, lookup, spreads, exemplars,
            base_slot + 1, n=grounding_n,
        )
        base_slot += len(grounding_rows)
        new_rows.extend(grounding_rows)
    else:
        print("  grounding slice already present — skipping (resume)")
    if not has_correction:
        correction_rows = generate_correction_examples(
            client, random.Random(args.seed + base_slot), cards, lookup, spreads, exemplars,
            base_slot + 1,
        )
        new_rows.extend(correction_rows)
    else:
        print("  correction slice already present — skipping (resume)")

    rows = rows + new_rows
    write_jsonl(out_path, rows)

    from collections import Counter
    counts = Counter(r["task_type"] for r in rows)
    pairs = sum(1 for r in rows if r["matched_pair_id"])
    print(f"  wrote {out_path}: {len(rows)} rows")
    print(f"  task types: {dict(counts)}")
    print(f"  matched safety pairs: {pairs // 2}")
    print(f"  critique_applied: {sum(1 for r in rows if r['critique_applied'])}")
    return 0


# ---------------------------------------------------------------------- w33 --


def run_w33(args: argparse.Namespace) -> int:
    print("== W3.3 anti-collapse ablation + diversity metrics")
    out_path = RAW_DIR / "generated.jsonl"
    rows = read_jsonl(out_path) if out_path.exists() else []
    if not rows:
        print("  no generated rows yet — run w32 first", file=sys.stderr)
        return 1
    memory = MemoryIndex(CACHE / "memory.jsonl")
    blacklist = SelfTighteningNGramBlacklist(profile_path=KB / "vn_register_profile.json")
    texts = [r["target_vi"] for r in rows if r.get("target_vi")]
    # feed corpus so forbidden_phrases / never-fires reflect real usage
    blacklist.update(texts)
    ablation = run_ablation(texts, memory, blacklist)

    main_texts = [
        r["target_vi"] for r in rows
        if r.get("target_vi") and not any(
            p in (r.get("provenance") or [])
            for p in ("safety_slice", "grounding_negative", "counter_sycophancy")
        )
    ]
    sample = main_texts[:200] if len(main_texts) >= 200 else main_texts
    rng = random.Random(args.seed)
    sample_random = rng.sample(main_texts, 200) if len(main_texts) >= 200 else main_texts
    windows = [
        distinct_n(main_texts[i:i + 100])
        for i in range(0, len(main_texts), 100)
    ]
    fresh = MemoryIndex()
    dups = 0
    for t in main_texts:
        if fresh.is_dup(t):
            dups += 1
        else:
            fresh.add(t)
    report = {
        "schema": "w33_ablation",
        "method": "none / memory-only / all-three (plan W3.3)",
        "caveat": (
            "The memory-only/all-three conditions filter against the PERSISTENT memory index "
            "that already contains the corpus itself (self-matches); their kept sets collapse "
            "to non-memory rows, so their distinct-2 is a size artifact, not a counterfactual "
            "ablation. The honest metrics are the corpus windows, the random 200-sample, and "
            "the corpus-internal replay dedup rate below."
        ),
        "ablation": ablation,
        "corpus_metrics": {
            "main_loop_rows": len(main_texts),
            "distinct2_first_200": distinct_n(sample),
            "distinct2_random_200": distinct_n(sample_random),
            "distinct2_per_100_windows": [round(w, 4) for w in windows],
            "distinct2_per_100_min": round(min(windows), 4) if windows else None,
            "corpus_internal_replay_dedup_rate": round(dups / len(main_texts), 4) if main_texts else None,
            "corpus_internal_replay_dedup_note": (
                "fresh MemoryIndex replayed over the corpus in file order; rate 0 means the "
                "cross-batch memory rejected all near-duplicates at generation time (dedup did "
                "its work); any rate > 0 would indicate a memory leak."
            ),
            "memory_index_size": memory.size,
            "blacklist_forbidden_phrases": len(blacklist.forbidden_phrases),
        },
        "acceptance": {
            "distinct2_random_200_ge_0.45": distinct_n(sample_random) >= 0.45,
            "distinct2_all_three_ge_0.45": ablation.get("distinct2_all_three", 0) >= 0.45,
            "blacklist_no_function_words": ablation["blacklist_never_fires_on_function_words"],
            "dedup_rate_below_0.4": dups / len(main_texts) < 0.4 if main_texts else True,
        },
    }
    out = DATA_DIR / "ablation_report.json"
    out.write_text(dumps_canonical(report) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    print(f"  distinct-2 (random 200): {report['corpus_metrics']['distinct2_random_200']} (floor 0.45)")
    print(f"  distinct-2 per-100 windows: {[round(w, 4) for w in windows]}")
    print(f"  corpus-internal replay dedup rate: {report['corpus_metrics']['corpus_internal_replay_dedup_rate']}")
    print(f"  blacklist: {len(blacklist.forbidden_phrases)} phrases, "
          f"never fires on function words: {ablation['blacklist_never_fires_on_function_words']}")
    return 0


# ------------------------------------------------------------------ w34 --

_VI_DIACRITICS = frozenset(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)
_MINHASH_P = (1 << 31) - 1
L1_LENGTH_BOUNDS = {"ngắn": (40, 320), "đầy_đủ": (60, 420)}
L1_REQUIRED_KEYS: Dict[str, type] = {
    "example_id": str, "question_vi": str, "target_vi": str, "reading_vi": str,
    "card_ids": list, "cards_used": list, "orientations": list, "task_type": str,
    "spread_id": str, "length_band": str, "register": str, "querent_context": str,
    "provenance": list, "prompt_slot": int,
}
L1_TASK_TYPES = {"explanation", "reading", "safety", "grounding", "correction"}
L1_LENGTH_BANDS = {"ngắn", "đầy_đủ"}
L1_REGISTERS = {"formal", "warm", "casual"}


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _vi_word_fraction(text: str) -> float:
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    if not words:
        return 0.0
    return sum(1 for w in words if any(c in _VI_DIACRITICS for c in w)) / len(words)


def _minhash_signature(text: str, k: int = 128, seed: int = 7) -> np.ndarray:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    shingles = {t[i : i + 5] for i in range(max(0, len(t) - 4))}
    if not shingles:
        return np.zeros(k, dtype=np.int64)
    vals = np.array([zlib.crc32(s.encode("utf-8")) for s in shingles], dtype=np.int64)
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _MINHASH_P, size=k).astype(np.int64)
    b = rng.randint(0, _MINHASH_P, size=k).astype(np.int64)
    return ((a[:, None] * vals[None, :] + b[:, None]) % _MINHASH_P).min(axis=1)


def _normalize(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def _cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    s = float(np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)))
    return (ma - mb) / s if s > 0 else 0.0


def _l1_check(row: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = []
    warns: List[str] = []
    for key, typ in L1_REQUIRED_KEYS.items():
        if key not in row or not isinstance(row[key], typ):
            reasons.append(f"schema:{key}")
    if row.get("task_type") not in L1_TASK_TYPES:
        reasons.append("schema:task_type")
    if row.get("length_band") not in L1_LENGTH_BANDS:
        reasons.append("schema:length_band")
    if row.get("register") not in L1_REGISTERS:
        reasons.append("schema:register")
    if isinstance(row.get("orientations"), list) and isinstance(row.get("cards_used"), list):
        if len(row["orientations"]) != len(row["cards_used"]):
            reasons.append("schema:orientation_count")
    text = _nfc(row.get("target_vi") or "")
    for key in ("question_vi", "target_vi", "reading_vi"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            reasons.append(f"schema:empty_{key}")
    if reasons:
        return False, reasons, warns
    if _vi_word_fraction(text) < 0.15:
        reasons.append("lang_id")
    if "khoẻ" in text:
        reasons.append("diacritic_health_typo")
    lo, hi = L1_LENGTH_BOUNDS[row["length_band"]]
    if not (lo <= len(text.split()) <= hi):
        reasons.append("length_bounds")
    low = text.lower()
    if any(c["name_en"].lower() not in low for c in row["cards_used"]):
        reasons.append("card_containment")
    if any(c["orientation"] == "reversed" for c in row["cards_used"]) and "ngược" not in low:
        reasons.append("orientation_no_nguoc")
    drawn = {c["name_en"].lower() for c in row["cards_used"]}
    for name in CANONICAL_NAMES:
        nl = name.lower()
        if nl not in drawn and low.count(nl) >= 2:
            warns.append(f"collision_warn:{name}")
            break
    return not reasons, reasons, warns


def _dedup(
    rows: Sequence[Dict[str, Any]],
    use_structural: bool = False,
    minhash_k: int = 128,
    minhash_j: float = 0.85,
    embed_cos: float = 0.92,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    embed = CharNGramEmbedding()
    stats = {"exact_sha256": 0, "structural": 0, "minhash_ge_0.85": 0, "embedding_ge_0.92": 0}
    kept: List[Dict[str, Any]] = []
    n = len(rows)
    sigs = np.zeros((n, minhash_k), dtype=np.int64)
    vecs = np.zeros((n, 4096), dtype=np.float32)
    lens = np.zeros(n, dtype=np.int64)
    hashes: set = set()
    structs: set = set()
    for r in rows:
        text = _nfc(r["target_vi"])
        h = hashlib.sha256(dumps_canonical(r).encode("utf-8")).hexdigest()
        if h in hashes:
            stats["exact_sha256"] += 1
            continue
        protected = bool(r.get("matched_pair_id"))
        if use_structural and not protected:
            key = (tuple(sorted(r["card_ids"])), tuple(r["orientations"]), r["task_type"],
                   r["spread_id"], r["length_band"], r["register"])
            if key in structs:
                stats["structural"] += 1
                continue
            structs.add(key)
        m = len(kept)
        lc = len(text.split())
        sig_c = _minhash_signature(text, minhash_k)
        v = _normalize(embed.vector(text))
        if not protected and m:
            win = np.where((lens[:m] * 1.2 >= lc) & (lens[:m] <= lc * 1.2))[0]
            if len(win):
                if (sigs[win] == sig_c).mean(axis=1).max() >= minhash_j:
                    stats["minhash_ge_0.85"] += 1
                    continue
                if (vecs[win] @ v).max() >= embed_cos:
                    stats["embedding_ge_0.92"] += 1
                    continue
        hashes.add(h)
        sigs[m] = sig_c
        vecs[m] = v
        lens[m] = lc
        kept.append(r)
    return kept, stats


def _student_model_path() -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots"
    snaps = sorted(hub.glob("*")) if hub.exists() else []
    if not snaps:
        raise SystemExit("Qwen3-1.7B not found in HF cache — layer 2 needs the student base model")
    return snaps[-1]


def _ifd_scores(
    rows: Sequence[Dict[str, Any]],
    model_path: Path,
    batch_size: int = 4,
    max_len: int = 512,
) -> List[float]:
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    pad = tok.pad_token_id
    scores: List[float] = []

    def batch_ll(encs: Sequence[Tuple[List[int], List[int]]], conditional: bool):
        seqs = [q + t if conditional else t for q, t in encs]
        L = min(max_len, max(len(s) for s in seqs))
        ids = torch.full((len(encs), L), pad, dtype=torch.long)
        labels = torch.full((len(encs), L), -100, dtype=torch.long)
        mask = torch.zeros((len(encs), L), dtype=torch.long)
        for j, (s, (q, t)) in enumerate(zip(seqs, encs)):
            s = s[:L]
            ids[j, : len(s)] = torch.tensor(s)
            if conditional:
                labels[j, len(q) : len(s)] = torch.tensor(s[len(q) :])
            else:
                labels[j, 1 : len(s)] = torch.tensor(s[1:])
            mask[j, : len(s)] = 1
        logits = model(input_ids=ids, attention_mask=mask).logits
        ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(len(encs), L - 1)
        valid = (labels[:, 1:] != -100) & mask[:, 1:].bool()
        return (ce * valid.float()).sum(1) / valid.sum(1).clamp(min=1)

    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            encs = []
            for r in batch:
                q = tok.encode(r["question_vi"], add_special_tokens=False)[: max_len // 2]
                t = tok.encode(r["target_vi"], add_special_tokens=False)[: max_len // 2]
                encs.append((q, t))
            cond_ll = batch_ll(encs, conditional=True)
            uncond_ll = batch_ll(encs, conditional=False)
            scores.extend((uncond_ll - cond_ll).tolist())
    del model
    return scores


def _l3_deita(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    if not rows:
        return [], 0
    texts = [r["target_vi"] for r in rows]
    words = np.array([len(t.split()) for t in texts], dtype=np.float64)
    ttr = np.array([len(set(t.split())) / max(1, len(t.split())) for t in texts], dtype=np.float64)
    sents = np.array([len(re.findall(r"[.!?…]", t)) + 1 for t in texts], dtype=np.float64)
    ifd = np.array([r["ifd_score"] for r in rows], dtype=np.float64)

    def z(x: np.ndarray) -> np.ndarray:
        s = x.std()
        return (x - x.mean()) / s if s > 0 else np.zeros_like(x)

    score = 0.5 * (z(np.log1p(words)) + z(ttr) + z(sents)) + 0.5 * z(ifd)
    order = np.argsort(-score, kind="stable")
    embed = CharNGramEmbedding()
    mat = np.zeros((len(rows), 4096), dtype=np.float32)
    kept_idx: List[int] = []
    m = 0
    for i in order:
        v = _normalize(embed.vector(texts[int(i)]))
        if m and (mat[:m] @ v).max() >= 0.92:
            continue
        mat[m] = v
        m += 1
        kept_idx.append(int(i))
    return [rows[i] for i in kept_idx], len(rows) - m


def _l4_judge(
    client: LLMClient,
    rows: Sequence[Dict[str, Any]],
    workers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    lock = threading.Lock()
    errors = 0

    def judge_one(i: int, r: Dict[str, Any]) -> Dict[str, Any]:
        msgs = build_rubric_messages(r["question_vi"], r["cards_used"], r["reading_vi"])
        try:
            return client.chat_json(msgs, max_tokens=400, temperature=0, model=client.model_sonnet)
        except LLMError:
            return {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(judge_one, i, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    axes = {"tone": [0, 0], "translationese": [0, 0], "faithfulness": [0, 0]}
    kept: List[Dict[str, Any]] = []
    for r, out in zip(rows, results):
        if not out or not out.get("decision"):
            with lock:
                errors += 1
            continue
        for ax, (p, f) in axes.items():
            v = out.get(ax) or {}
            if v.get("pass"):
                axes[ax][0] += 1
            else:
                axes[ax][1] += 1
        if out.get("decision") == "pass":
            kept.append(r)
    report = {
        "judge": client.model_sonnet,
        "gating_axes": ["tone", "translationese", "faithfulness"],
        "excluded_axes": ["orientation"],
        "per_axis": {ax: {"pass": p, "fail": f} for ax, (p, f) in axes.items()},
        "decision_pass": len(kept),
        "decision_fail": len(rows) - len(kept) - errors,
        "judge_errors": errors,
    }
    report["systematic_defect"] = any(
        f / max(1, p + f) > 0.6 for p, f in axes.values()
    )
    return kept, report


def run_w34(args: argparse.Namespace) -> int:
    print("== W3.4 four-layer filter (L1 programmatic / L2 IFD / L3 Deita / L4 judge)")
    load_env()
    from collections import Counter

    in_path = RAW_DIR / "generated.jsonl"
    rows = read_jsonl(in_path) if in_path.exists() else []
    if not rows:
        print("  no generated rows — run w32 first", file=sys.stderr)
        return 1
    print(f"  input: {len(rows)} rows")

    l1_ok: List[Dict[str, Any]] = []
    l1_fail: Counter = Counter()
    l1_warns: Counter = Counter()
    for r in rows:
        ok, reasons, warns = _l1_check(r)
        if ok:
            l1_ok.append(r)
        else:
            l1_fail.update(reasons)
        l1_warns.update(warns)
    l1_kept, l1_dedup = _dedup(l1_ok, use_structural=False)
    print(f"  L1 programmatic: {len(rows)} -> {len(l1_ok)} "
          f"(drops {len(rows) - len(l1_ok)}: {dict(l1_fail.most_common(6))})")
    print(f"  L1 dedup: {len(l1_ok)} -> {len(l1_kept)} ({l1_dedup})")
    print(f"  L1 collision warns (deferred to L4 faithfulness): {dict(l1_warns.most_common(5))}")
    if len(l1_kept) < 2:
        print("  too few rows after L1", file=sys.stderr)
        return 1

    model_path = Path(args.ifd_model) if args.ifd_model else _student_model_path()
    print(f"  L2 IFD model: {model_path.name}")
    scores = _ifd_scores(l1_kept, model_path, batch_size=args.ifd_batch)
    for r, s in zip(l1_kept, scores):
        r["ifd_score"] = round(float(s), 4)
    order = sorted(range(len(l1_kept)), key=lambda i: (-l1_kept[i]["ifd_score"], l1_kept[i]["example_id"]))
    n_keep = max(1, int(round(len(l1_kept) * args.ifd_keep)))
    l2_kept = [l1_kept[i] for i in order[:n_keep]]
    l2_rejected = [l1_kept[i] for i in order[n_keep:]]
    d = _cohens_d([r["ifd_score"] for r in l2_kept], [r["ifd_score"] for r in l2_rejected]) if l2_rejected else None
    print(f"  L2 IFD: {len(l1_kept)} -> {len(l2_kept)} (keep {args.ifd_keep}) "
          f"cohen's d kept-vs-rejected: {round(d, 3) if d is not None else 'n/a'}")
    l2_report = {
        "model": "Qwen/Qwen3-1.7B",
        "keep_fraction": args.ifd_keep,
        "kept_mean_ifd": round(float(np.mean([r["ifd_score"] for r in l2_kept])), 4),
        "rejected_mean_ifd": round(float(np.mean([r["ifd_score"] for r in l2_rejected])), 4) if l2_rejected else None,
        "cohens_d": None if d is None else round(float(d), 4),
        "measurably_different": d is not None and abs(d) >= 0.2,
    }

    l3_kept, l3_dropped = _l3_deita(l2_kept)
    print(f"  L3 Deita: {len(l2_kept)} -> {len(l3_kept)} (diversity-greedy dropped {l3_dropped})")

    if args.skip_l4:
        l4_kept, l4_report = l3_kept, {"skipped": True, "judge": None}
        print("  L4 skipped (--skip-l4)")
    else:
        client = LLMClient(seed=args.seed)
        if not client.model_sonnet:
            print("  LLM_MODEL_SONNET unset — L4 requires the calibrated independent judge",
                  file=sys.stderr)
            return 1
        print(f"  L4 judge: {client.model_sonnet} over {len(l3_kept)} rows (workers={args.workers})")
        l4_kept, l4_report = _l4_judge(client, l3_kept, workers=args.workers)
        print(f"  L4 judge: {len(l3_kept)} -> {len(l4_kept)} pass "
              f"per-axis={l4_report.get('per_axis')} errors={l4_report.get('judge_errors')}")

    core = l4_kept[: args.max_core]
    core_ids = {r["example_id"] for r in core}
    bulk = [r for r in l2_kept if r["example_id"] not in core_ids]
    for r in core:
        r["provenance"] = list(r.get("provenance") or []) + ["filtered_core"]
    for r in bulk:
        r["provenance"] = list(r.get("provenance") or []) + ["filtered_bulk"]

    write_jsonl(DATA_DIR / "filtered_core.jsonl", core)
    write_jsonl(DATA_DIR / "filtered_bulk.jsonl", bulk)

    report = {
        "schema": "w34_filter",
        "input_rows": len(rows),
        "layers": {
            "l1_programmatic": {"kept": len(l1_ok), "dropped": len(rows) - len(l1_ok),
                                "reasons": dict(l1_fail.most_common()),
                                "collision_warns_deferred_to_l4_faithfulness": dict(l1_warns.most_common())},
            "l1_dedup": {"kept": len(l1_kept), **l1_dedup},
            "l2_ifd": {"kept": len(l2_kept), "dropped": len(l2_rejected), **l2_report},
            "l3_deita": {"kept": len(l3_kept), "dropped": l3_dropped,
                         "method": "complexity(z-log-words+z-TTR+z-sentences)+quality(IFD) score-first, "
                                   "diversity-greedy cosine<0.92"},
            "l4_judge": {"kept": len(l4_kept), **l4_report},
        },
        "tiers": {"core_rows": len(core), "bulk_rows": len(bulk)},
        "acceptance": {
            "core_ge_3000": len(core) >= 3000,
            "bulk_ge_8000": len(bulk) >= 8000,
            "l4_calibrated_judge": not args.skip_l4,
            "ifd_sets_differ": l2_report["measurably_different"],
        },
        "outputs": ["datasets/filtered_core.jsonl", "datasets/filtered_bulk.jsonl"],
    }
    out = DATA_DIR / "filter_report.json"
    out.write_text(dumps_canonical(report) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    print(f"  tiers: core={len(core)} bulk={len(bulk)}")
    return 0


# ------------------------------------------------------------------ w35 --


def _row_card_combos(row: Dict[str, Any]) -> set:
    return set(zip(row["card_ids"], row["orientations"]))


def run_w35(args: argparse.Namespace) -> int:
    print("== W3.5 dedup cascade + coverage analysis")
    core = read_jsonl(DATA_DIR / "filtered_core.jsonl") if (DATA_DIR / "filtered_core.jsonl").exists() else []
    bulk = read_jsonl(DATA_DIR / "filtered_bulk.jsonl") if (DATA_DIR / "filtered_bulk.jsonl").exists() else []
    if not core and not bulk:
        print("  no filtered tiers — run w34 first", file=sys.stderr)
        return 1
    core_ids = {r["example_id"] for r in core}
    combined = core + bulk
    print(f"  combined: {len(combined)} rows (core={len(core)} bulk={len(bulk)})")

    kept, cascade = _dedup(combined, use_structural=True)
    kept_ids = {r["example_id"] for r in kept}
    core_kept = [r for r in core if r["example_id"] in kept_ids]

    all_combos = {combo for r in combined for combo in _row_card_combos(r)}
    core_combos = {combo for r in core_kept for combo in _row_card_combos(r)}
    missing_cards = sorted(all_combos - core_combos)

    spreads = load_spreads()
    spread_ids = [s["spread_id"] for s in spreads]
    missing_spreads = [sid for sid in spread_ids if not any(r["spread_id"] == sid for r in core_kept)]

    pairs_per_cat: Dict[str, Dict[str, int]] = {}
    for r in core_kept:
        if r.get("matched_pair_id"):
            d = pairs_per_cat.setdefault(r["safety_category"], {})
            d[r["matched_pair_id"]] = d.get(r["matched_pair_id"], 0) + 1
    safety_pairs = {cat: sum(1 for c in m.values() if c >= 2) for cat, m in pairs_per_cat.items()}
    safety_ok = all(safety_pairs.get(cat, 0) >= 5 for cat in SAFETY_CATEGORIES)

    canonical = "\n".join(
        dumps_canonical(r) for r in sorted(kept, key=lambda r: r["example_id"])
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (DATA_DIR / "DATASET_HASH.txt").write_text(digest + "\n", encoding="utf-8")

    report = {
        "schema": "w35_coverage",
        "rows_in": len(combined),
        "rows_after_dedup": len(kept),
        "dedup_cascade": cascade,
        "core_rows_after_dedup": len(core_kept),
        "coverage": {
            "card_orientation_in_core": {
                "expected": len(all_combos),
                "present": len(core_combos),
                "missing": [f"{cid}:{o}" for cid, o in missing_cards],
            },
            "spreads_in_core": {
                "expected": len(spread_ids),
                "present": len(spread_ids) - len(missing_spreads),
                "missing": missing_spreads,
            },
            "safety_pairs_per_category_in_core": safety_pairs,
        },
        "acceptance": {
            "all_card_orientation_in_core": not missing_cards,
            "all_spreads_in_core": not missing_spreads,
            "safety_pairs_ge_5_each": safety_ok,
            "dedup_removed_something": sum(cascade.values()) > 0,
        },
        "hash": digest,
        "hash_file": "datasets/DATASET_HASH.txt",
    }
    out = DATA_DIR / "coverage_report.json"
    out.write_text(dumps_canonical(report) + "\n", encoding="utf-8")
    print(f"  dedup cascade: {cascade}")
    print(f"  missing (card,orientation) in core: {len(missing_cards)}")
    print(f"  missing spreads in core: {missing_spreads}")
    print(f"  safety pairs per category: {safety_pairs}")
    print(f"  wrote {out} and datasets/DATASET_HASH.txt ({digest[:16]}...)")
    ok = not missing_cards and not missing_spreads and safety_ok
    return 0 if ok else 1


# ------------------------------------------------------------------ w36 --

ANCHOR_CARDS = [
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
    "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
]


def _build_anchors() -> List[Dict[str, Any]]:
    src = ROOT / "data" / "vietnamese" / "Tarot-Vietnamese-API" / "data.txt"
    records = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in records:
        name = rec.get("name", "")
        if not name.startswith("Lá bài "):
            continue
        rest = name[len("Lá bài "):].strip()
        orientation = "reversed" if rest.endswith(" ngược") else "upright"
        name_en = rest[: -len(" ngược")].strip() if orientation == "reversed" else rest
        if name_en == "The Wheel of Fortune":
            name_en = "Wheel of Fortune"
        if name_en in NAME_TO_ID:
            by_key[(name_en, orientation)] = rec
    anchors: List[Dict[str, Any]] = []
    for i, name_en in enumerate(ANCHOR_CARDS):
        for orientation in ("upright", "reversed"):
            rec = by_key.get((name_en, orientation))
            if not rec:
                print(f"  anchor missing: {name_en} {orientation}", file=sys.stderr)
                continue
            anchors.append({
                "anchor_id": f"anchor_{i + 1:02d}_{orientation[0]}",
                "card_id": NAME_TO_ID[name_en],
                "name_en": name_en,
                "orientation": orientation,
                "prose_vi": _nfc(rec.get("title_main", "")),
                "source": "tarot_vietnamese_api",
                "provenance": ["anchor_authentic"],
            })
    return anchors


def run_w36(args: argparse.Namespace) -> int:
    print("== W3.6 stratified splits + anchor isolation")
    core = read_jsonl(DATA_DIR / "filtered_core.jsonl") if (DATA_DIR / "filtered_core.jsonl").exists() else []
    bulk = read_jsonl(DATA_DIR / "filtered_bulk.jsonl") if (DATA_DIR / "filtered_bulk.jsonl").exists() else []
    combined = core + bulk
    if not combined:
        print("  no filtered tiers — run w34 first", file=sys.stderr)
        return 1
    print(f"  combined rows: {len(combined)}")

    anchor_dir = DATA_DIR / "anchor"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    anchors = _build_anchors()
    if len(anchors) != 30:
        print(f"  expected 30 anchors, got {len(anchors)}", file=sys.stderr)
        return 1
    write_jsonl(anchor_dir / "anchor_readings.jsonl", anchors)
    print(f"  anchors: {len(anchors)} -> datasets/anchor/anchor_readings.jsonl")

    rng = random.Random(args.seed)
    pair_units: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    units: List[List[Dict[str, Any]]] = []
    for r in combined:
        if r.get("matched_pair_id"):
            pair_units.setdefault((r["safety_category"], r["matched_pair_id"]), []).append(r)
        else:
            units.append([r])
    for g in pair_units.values():
        units.append(g)

    strata: Dict[Tuple[str, str, str], List[List[Dict[str, Any]]]] = {}
    for unit in units:
        r0 = unit[0]
        strata.setdefault((r0["task_type"], r0["length_band"], r0["register"]), []).append(unit)

    splits: Dict[str, str] = {}
    for key in sorted(strata):
        u = strata[key]
        rng.shuffle(u)
        n = len(u)
        n_test = int(round(n * 0.07))
        n_val = int(round(n * 0.08))
        n_train = n - n_test - n_val
        for unit in u[:n_train]:
            for r in unit:
                splits[r["example_id"]] = "train"
        for unit in u[n_train:n_train + n_val]:
            for r in unit:
                splits[r["example_id"]] = "val"
        for unit in u[n_train + n_val:]:
            for r in unit:
                splits[r["example_id"]] = "test"

    all_combos = {combo for r in combined for combo in _row_card_combos(r)}
    train_rows = [r for r in combined if splits[r["example_id"]] == "train"]
    train_combos = {combo for r in train_rows for combo in _row_card_combos(r)}
    missing = sorted(all_combos - train_combos)
    for cid, o in missing:
        for unit in units:
            if splits[unit[0]["example_id"]] == "train":
                continue
            if (cid, o) in {combo for r in unit for combo in _row_card_combos(r)}:
                for r in unit:
                    splits[r["example_id"]] = "train"
                break
    train_rows = [r for r in combined if splits[r["example_id"]] == "train"]
    still_missing = sorted(all_combos - {combo for r in train_rows for combo in _row_card_combos(r)})
    if still_missing:
        print(f"  cannot cover (card,orientation) in train: {still_missing}", file=sys.stderr)
        return 1

    (DATA_DIR / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    train = {i for i, s in splits.items() if s == "train"}
    val = {i for i, s in splits.items() if s == "val"}
    test = {i for i, s in splits.items() if s == "test"}
    test_task_types = {r["task_type"] for r in combined if splits[r["example_id"]] == "test"}
    no_overlap = not (train & val or train & test or val & test)
    anchors_isolated = not any(a["anchor_id"] in splits for a in anchors)
    test_size_ok = len(test) >= 500
    test_covers = test_task_types >= L1_TASK_TYPES

    stats = {
        "schema": "w36_splits",
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "test_task_types": sorted(test_task_types),
        "acceptance": {
            "no_overlap": no_overlap,
            "anchors_isolated": anchors_isolated,
            "every_card_orientation_in_train": not still_missing,
            "test_ge_500": test_size_ok,
            "test_covers_all_task_types": test_covers,
        },
        "anchor_file": "datasets/anchor/anchor_readings.jsonl",
    }
    (DATA_DIR / "split_stats.json").write_text(dumps_canonical(stats) + "\n", encoding="utf-8")
    print(f"  splits: {stats['sizes']} test task types: {stats['test_task_types']}")
    print(f"  wrote datasets/splits.json and datasets/split_stats.json")
    ok = no_overlap and anchors_isolated and not still_missing and test_size_ok and test_covers
    return 0 if ok else 1


# ------------------------------------------------------------------- main --


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p32 = sub.add_parser("w32", help="generate the dataset")
    p32.add_argument("--limit", type=int, default=40, help="examples to generate (default 40)")
    p32.add_argument("--seed", type=int, default=42)
    p32.add_argument("--workers", type=int, default=4, help="parallel generation workers")
    p32.add_argument("--critique-fraction", type=float, default=0.5,
                     help="fraction of readings getting critique-and-revise (default 0.5)")
    p32.add_argument("--safety-pairs", type=int, default=2, help="pairs per safety category")
    p32.add_argument("--grounding", type=int, default=6, help="grounding negatives")
    p32.set_defaults(fn=run_w32)
    p33 = sub.add_parser("w33", help="anti-collapse ablation + diversity")
    p33.add_argument("--seed", type=int, default=42)
    p33.set_defaults(fn=run_w33)
    p34 = sub.add_parser("w34", help="4-layer filter (L1 programmatic / L2 IFD / L3 Deita / L4 judge)")
    p34.add_argument("--ifd-keep", type=float, default=0.85, help="IFD keep fraction (default 0.85)")
    p34.add_argument("--max-core", type=int, default=5000, help="core tier cap (default 5000)")
    p34.add_argument("--workers", type=int, default=8, help="judge workers (default 8)")
    p34.add_argument("--seed", type=int, default=42)
    p34.add_argument("--ifd-model", type=str, default="",
                     help="student base model path (default: HF cache Qwen3-1.7B)")
    p34.add_argument("--ifd-batch", type=int, default=4)
    p34.add_argument("--skip-l4", action="store_true", help="stop after L3 (debug)")
    p34.set_defaults(fn=run_w34)
    p35 = sub.add_parser("w35", help="dedup cascade + coverage + DATASET_HASH")
    p35.add_argument("--seed", type=int, default=42)
    p35.set_defaults(fn=run_w35)
    p36 = sub.add_parser("w36", help="stratified 85/8/7 splits + anchor isolation")
    p36.add_argument("--seed", type=int, default=42)
    p36.set_defaults(fn=run_w36)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
