"""Validator 2 — orientation consistency (happy + failure + edge fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import read_jsonl  # noqa: E402
from tfvn.validators import validate_orientation_consistency  # noqa: E402


@pytest.fixture(scope="module")
def spine():
    path = ROOT / "kb/english_spine.jsonl"
    if not path.exists():
        pytest.skip("spine not built")
    return read_jsonl(path)


def test_orientation_happy(spine):
    text = (
        "The Tower ngược báo hiệu sự sụp đổ trì hoãn và hỗn loạn nội tâm; "
        "đây không phải năng lượng upright vững chắc."
    )
    draw = [{"card_id": 16, "orientation": "reversed", "name_en": "The Tower"}]
    result = validate_orientation_consistency(text, draw, spine)
    assert result["ok"] is True


def test_orientation_fails_reversed_with_upright_keywords(spine):
    fool_up = next(
        r for r in spine if r["name_en"] == "The Fool" and r["orientation"] == "upright"
    )
    kws = fool_up["keyword_atoms_en"][:4]
    text = (
        f"The Fool reversed still shows {kws[0]}, {kws[1]}, {kws[2]}, and {kws[3]} "
        f"as if brand-new beginnings."
    )
    draw = [{"card_id": 0, "orientation": "reversed", "name_en": "The Fool"}]
    result = validate_orientation_consistency(text, draw, spine)
    assert result["ok"] is False
    assert result["violations"]


def test_orientation_fails_named_as_upright(spine):
    text = "The Magician xuôi mang lại sự làm chủ tuyệt đối."
    draw = [{"card_id": 1, "orientation": "reversed", "name_en": "The Magician"}]
    result = validate_orientation_consistency(text, draw, spine)
    assert result["ok"] is False


def test_orientation_ignores_upright_draws(spine):
    """Draws that are not reversed are skipped entirely."""
    text = "The Fool xuôi mở đầu hành trình."
    draw = [{"card_id": 0, "orientation": "upright", "name_en": "The Fool"}]
    result = validate_orientation_consistency(text, draw, spine)
    assert result["ok"] is True


def test_orientation_fallback_without_kb_rows():
    """No KB rows -> fall back to name + 'xuôi'/'upright' proximity check."""
    text = "The Tower xuôi báo hiệu sự sụp đổ."
    draw = [{"card_id": 16, "orientation": "reversed", "name_en": "The Tower"}]
    result = validate_orientation_consistency(text, draw, kb_rows=None)
    assert result["ok"] is False
    assert result["violations"]


def test_orientation_short_keywords_skipped():
    """Keyword atoms shorter than 4 chars are ignored (noise guard)."""
    rows = [
        {"card_id": 0, "orientation": "upright", "keyword_atoms_en": ["joy", "new beginnings"]}
    ]
    text = "joy and new beginnings together"
    draw = [{"card_id": 0, "orientation": "reversed", "name_en": "The Fool"}]
    # 1 hit ('new beginnings') < 2 required -> no keyword violation; no name cue
    result = validate_orientation_consistency(text, draw, rows)
    assert result["ok"] is True
