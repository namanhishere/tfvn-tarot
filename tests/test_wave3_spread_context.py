"""Regression tests: spread/querent-context domain coupling (Wave 3).

Covers the fix for the W3.2 defect where a domain-specific spread (relationship
spreads, Decision Spread) was sampled independently of the querent context,
producing e.g. money questions on love-triangle spreads (82% of domain-spread
rows mismatched). See scripts/build_wave3.py: SPREAD_CONTEXT_ALLOW.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_wave3 as w3  # noqa: E402

ALL_CTX = {c for c, _ in w3.CONTEXTS}


def _valid_row() -> dict:
    text = (
        "Lá bài The Fool xuôi trong trải bài hôm nay gợi ý rằng bạn đang đứng "
        "trước một khởi đầu mới, nơi sự ngây thơ và niềm tin sẽ dẫn lối cho "
        "những bước đi tiếp theo. Tinh thần cởi mở giúp bạn đón nhận cơ hội "
        "mà không bị ràng buộc bởi nỗi sợ hãi hay định kiến cũ. Hãy lắng nghe "
        "trực giác và cho phép bản thân thử nghiệm những điều chưa từng làm "
        "trước đây, bởi vì hành trình phía trước tuy có phần bất định nhưng "
        "lại chứa đựng tiềm năng phát triển rất lớn."
    )
    return {
        "example_id": "w32_999999",
        "question_vi": "Ngày mai của tôi sẽ thế nào ạ?",
        "target_vi": text,
        "reading_vi": text,
        "card_ids": [0],
        "cards_used": [
            {"card_id": 0, "name_en": "The Fool",
             "orientation": "upright", "polarity_axis": "xuôi"}
        ],
        "orientations": ["upright"],
        "task_type": "reading",
        "spread_id": "spread_single",
        "length_band": "đầy_đủ",
        "register": "warm",
        "querent_context": "love",
        "provenance": ["w32_generated"],
        "prompt_slot": 1,
    }


def test_spread_allowed_contexts_domain_restricted():
    assert w3.spread_allowed_contexts("spread_relationship1") == ("love",)
    assert w3.spread_allowed_contexts("spread_relationship2") == ("love",)
    assert w3.spread_allowed_contexts("spread_relationship3") == ("love",)
    assert w3.spread_allowed_contexts("spread_decision") == ("decision",)


def test_spread_allowed_contexts_generic_all():
    for sid in ("spread_single", "spread_three", "spread_celticcross",
                "spread_astrological", "spread_threedragons"):
        assert set(w3.spread_allowed_contexts(sid)) == ALL_CTX


def test_sample_context_never_mismatches_domain_spreads():
    for sid in ("spread_relationship1", "spread_relationship2",
                "spread_relationship3", "spread_decision"):
        allowed = set(w3.spread_allowed_contexts(sid))
        for seed in range(300):
            ctx_id, _ = w3.sample_context(random.Random(seed), sid)
            assert ctx_id in allowed, f"{sid} drew mismatched context {ctx_id}"


def test_sample_context_generic_covers_all_contexts():
    seen = set()
    for seed in range(600):
        ctx_id, _ = w3.sample_context(random.Random(seed), "spread_single")
        seen.add(ctx_id)
    assert seen == ALL_CTX


def test_l1_check_rejects_spread_context_mismatch():
    row = _valid_row()
    row["spread_id"] = "spread_relationship3"
    row["querent_context"] = "money"
    ok, reasons, _ = w3._l1_check(row)
    assert not ok
    assert "spread_context_mismatch" in reasons


def test_l1_check_accepts_matched_domain_row():
    row = _valid_row()
    row["spread_id"] = "spread_relationship3"
    row["querent_context"] = "love"
    ok, reasons, _ = w3._l1_check(row)
    assert ok, reasons
