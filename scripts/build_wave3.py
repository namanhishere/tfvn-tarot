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
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.llm_client import LLMClient, LLMError, load_env  # noqa: E402
from tfvn.serialise import dumps_canonical, read_jsonl, write_jsonl  # noqa: E402
from tfvn.w3_anticollapse import (  # noqa: E402
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

    # Special slices.
    base_slot = slot
    safety_rows = generate_safety_examples(
        client, random.Random(args.seed + base_slot), cards, lookup, spreads, base_slot + 1,
        model_sonnet=model_sonnet, per_category=safety_n,
    )
    base_slot += len(safety_rows)
    grounding_rows = generate_grounding_examples(
        client, random.Random(args.seed + base_slot), cards, lookup, spreads, exemplars,
        base_slot + 1, n=grounding_n,
    )
    base_slot += len(grounding_rows)
    correction_rows = generate_correction_examples(
        client, random.Random(args.seed + base_slot), cards, lookup, spreads, exemplars,
        base_slot + 1,
    )

    rows = rows + safety_rows + grounding_rows + correction_rows
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
    ablation = run_ablation(texts, memory, blacklist)
    sample = texts[:200] if len(texts) >= 200 else texts
    ablation["distinct2_heldout_200"] = distinct_n(sample)
    ablation["n_rows"] = len(rows)
    report = {
        "schema": "w33_ablation",
        "method": "none / memory-only / all-three (plan W3.3)",
        "ablation": ablation,
        "acceptance": {
            "distinct2_all_three_ge_0.45": ablation.get("distinct2_all_three", 0) >= 0.45,
            "blacklist_no_function_words": ablation["blacklist_never_fires_on_function_words"],
        },
    }
    out = DATA_DIR / "ablation_report.json"
    out.write_text(dumps_canonical(report) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    print(f"  distinct-2: none={ablation['distinct2_none']} memory={ablation['distinct2_memory_only']} "
          f"all-three={ablation['distinct2_all_three']} (floor 0.45)")
    print(f"  dedup-rate probe: {memory.dedup_rate_probe(texts[:200])}")
    return 0


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
    p33.set_defaults(fn=run_w33)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
