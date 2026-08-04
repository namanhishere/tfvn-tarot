"""Validator 1 — card-name containment (happy + failure + edge fixtures)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.validators import (  # noqa: E402
    extract_card_names,
    validate_card_name_containment,
)


@pytest.fixture(scope="module")
def whitelist():
    path = ROOT / "kb/card_name_whitelist.json"
    if not path.exists():
        pytest.skip("whitelist not built")
    return json.loads(path.read_text(encoding="utf-8"))


def test_containment_happy(whitelist):
    text = (
        "Khi The Magician xuất hiện cùng Three of Cups, "
        "bạn đang ở giai đoạn sáng tạo và vui vẻ."
    )
    result = validate_card_name_containment(
        text, whitelist, required_names=["The Magician", "Three of Cups"]
    )
    assert result["ok"] is True
    assert "The Magician" in result["mentioned"]
    assert "Three of Cups" in result["mentioned"]


def test_containment_fails_on_hallucinated_card(whitelist):
    text = "The Magician opens the reading, then The Algorithm appears as a major force."
    result = validate_card_name_containment(text, whitelist, required_names=["The Magician"])
    assert result["ok"] is False
    assert "The Algorithm" in result["hallucinated_like"]


def test_containment_fails_missing_required(whitelist):
    text = "Only The Fool is discussed here."
    result = validate_card_name_containment(text, whitelist, required_names=["The Fool", "Death"])
    assert result["ok"] is False
    assert "Death" in result["missing_required"]


def test_containment_default_whitelist_loads_from_kb():
    """No whitelist passed -> load_whitelist() reads kb/card_name_whitelist.json."""
    result = validate_card_name_containment("The Fool and The Magician appear.")
    assert result["ok"] is True
    assert "The Fool" in result["mentioned"]


def test_containment_falls_back_when_kb_missing(monkeypatch):
    """load_whitelist raises FileNotFoundError -> in-memory canonical fallback."""

    def _boom():
        raise FileNotFoundError

    monkeypatch.setattr("tfvn.validators.load_whitelist", _boom)
    result = validate_card_name_containment("The Fool and The Magician appear.")
    assert result["ok"] is True
    assert "The Magician" in result["mentioned"]


def test_containment_unresolvable_required_name(whitelist):
    """required_names entry that is not in the alias table -> KeyError path."""
    result = validate_card_name_containment(
        "Only The Fool is discussed here.",
        whitelist,
        required_names=["The Algorithm"],
    )
    assert result["ok"] is False
    assert "The Algorithm" in result["missing_required"]


def test_extract_overlapping_names_not_double_counted(whitelist):
    """Alias 'The Hierophant or Pope' overlaps canonical 'The Hierophant'."""
    mentioned = extract_card_names("The Hierophant or Pope appears.", whitelist)
    assert mentioned.count("The Hierophant") == 1


def test_containment_common_phrase_not_hallucination(whitelist):
    """'The Past' is a reading-phrase, not a hallucinated major."""
    result = validate_card_name_containment("The Past matters; The Fool is drawn.", whitelist)
    assert result["ok"] is True
    assert "The Past" not in result["hallucinated_like"]
