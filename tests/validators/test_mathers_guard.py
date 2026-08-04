"""Validator 5 — Mathers numeric-join guard (happy + failure fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.validators import (  # noqa: E402
    mathers_numeric_join_is_type_error,
    validate_mathers_join_guard,
)


def test_mathers_guard_happy():
    rec = {
        "name_source": "Deuce of Cups",
        "meaning_upright": "Love, Attachment",
        "meaning_reversed": "Crossed desires, Obstacles",
        "source": "mathers_1888",
    }
    result = validate_mathers_join_guard(rec)
    assert result["ok"] is True


def test_mathers_guard_fails_with_numeric_id():
    rec = {
        "name_source": "Deuce of Cups",
        "meaning_upright": "Love",
        "meaning_reversed": "Obstacles",
        "source": "mathers_1888",
        "card_id": 48,  # banned
    }
    result = validate_mathers_join_guard(rec)
    assert result["ok"] is False
    assert "card_id" in result["banned_keys_present"]


def test_mathers_guard_fails_nested_join_dict():
    """A numeric join key smuggled under a nested 'join' dict also fails."""
    rec = {
        "name_source": "Deuce of Cups",
        "meaning_upright": "Love",
        "meaning_reversed": "Obstacles",
        "source": "mathers_1888",
        "join": {"id": 48},
    }
    result = validate_mathers_join_guard(rec)
    assert result["ok"] is False
    assert "join.numeric" in result["banned_keys_present"]


def test_mathers_numeric_join_raises_type_error():
    rec = {
        "name_source": "The Juggler",
        "meaning_upright": "Will",
        "meaning_reversed": "Cunning",
        "source": "mathers_1888",
        "numeric_id": 1,
    }
    with pytest.raises(TypeError):
        mathers_numeric_join_is_type_error(rec)
