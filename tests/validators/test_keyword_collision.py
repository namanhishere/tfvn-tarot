"""Validator 3 — keyword collision (happy + failure fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import read_jsonl  # noqa: E402
from tfvn.validators import validate_keyword_collision  # noqa: E402


@pytest.fixture(scope="module")
def spine():
    path = ROOT / "kb/english_spine.jsonl"
    if not path.exists():
        pytest.skip("spine not built")
    return read_jsonl(path)


def test_keyword_collision_happy(spine):
    text = (
        "The Magician emphasises skill, willpower, and focused intention "
        "in this single-card draw."
    )
    draw = [{"card_id": 1, "orientation": "upright", "name_en": "The Magician"}]
    result = validate_keyword_collision(text, draw, spine)
    assert result["ok"] is True


def test_keyword_collision_fails_undrawn_card_keywords(spine):
    death = next(
        r for r in spine if r["name_en"] == "Death" and r["orientation"] == "upright"
    )
    kws = [k for k in death["keyword_atoms_en"] if len(k) >= 5][:3]
    if len(kws) < 2:
        pytest.skip("not enough distinctive death keywords")
    text = f"Unexpected themes of {kws[0]} and {kws[1]} dominate the narrative."
    draw = [{"card_id": 1, "orientation": "upright", "name_en": "The Magician"}]
    result = validate_keyword_collision(text, draw, spine)
    if result["ok"]:
        text = " ".join(death["keyword_atoms_en"][:5])
        result = validate_keyword_collision(text, draw, spine)
    assert result["ok"] is False
    assert any(c["name_en"] == "Death" for c in result["collisions"])
