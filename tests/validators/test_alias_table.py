"""Validator 4 — alias table totality + function (happy + failure fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.aliases import ALIAS_TABLE, build_alias_table  # noqa: E402
from tfvn.validators import validate_alias_table  # noqa: E402


def test_alias_table_happy():
    result = validate_alias_table()
    assert result["ok"] is True
    assert result["n_canonical"] == 78
    assert ALIAS_TABLE["the juggler"] == "The Magician"
    assert ALIAS_TABLE["the foolish man"] == "The Fool"
    assert ALIAS_TABLE["fortitude"] == "Strength"
    assert ALIAS_TABLE["the last judgment"] == "Judgement"
    assert ALIAS_TABLE["the universe"] == "The World"
    assert ALIAS_TABLE["deuce of cups"] == "Two of Cups"
    assert ALIAS_TABLE["knave of sceptres"] == "Page of Wands"


def test_alias_table_fails_when_incomplete():
    broken = build_alias_table()
    del broken["the fool"]
    result = validate_alias_table(broken)
    assert result["ok"] is False


def test_alias_table_fails_bad_self_map():
    """A canonical self-map whose value is a different card is a broken table."""
    broken = build_alias_table()
    broken["the fool"] = "The Magician"
    result = validate_alias_table(broken)
    assert result["ok"] is False
    assert "bad self-map" in result["error"]


def test_alias_table_fails_non_canonical_target():
    broken = build_alias_table()
    broken["shadow realm"] = "Not A Card"
    result = validate_alias_table(broken)
    assert result["ok"] is False
