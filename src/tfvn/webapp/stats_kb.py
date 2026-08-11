"""KB / spreads / anchor stats for the dataset viewer (todo A.3 helper module).

* KB stats from ``kb/cards.jsonl``: per-arcana/suit/orientation/
  vi_provenance/polarity_axis/vi_orientation_attribution counts, meaning_en
  and meaning_vi char-length min/mean/max, and per-domain_vi-key non-empty
  coverage split by orientation.
* Spread stats from ``kb/spreads.jsonl`` (per-spread_id summary plus
  cards_drawn / difficulty distributions).
* Anchor stats from ``datasets/anchor/anchor_readings.jsonl``: per-card_id +
  per-orientation counts only — anchor rows carry neither ``task_type`` nor
  ``cards_used``, so no other field is derived for them.

Consumed by ``stats.py``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .stats_sft import _count_field, _read_jsonl


def _char_length_stats(
    rows: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    lengths = [
        len(row[field]) for row in rows if isinstance(row.get(field), str) and row[field]
    ]
    if not lengths:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
    }


def _domain_vi_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    keys = sorted(
        {
            key
            for row in rows
            if isinstance(row.get("domain_vi"), dict)
            for key in row["domain_vi"]
        }
    )
    coverage: dict[str, dict[str, int]] = {}
    for key in keys:
        per_orientation: Counter[str] = Counter()
        for row in rows:
            value = (row.get("domain_vi") or {}).get(key)
            if isinstance(value, str) and value.strip():
                per_orientation[str(row.get("orientation"))] += 1
        coverage[key] = {
            orientation: per_orientation[orientation]
            for orientation in sorted(per_orientation)
        }
    return coverage


def _kb_stats(cards_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(cards_path)
    return {
        "total_rows": len(rows),
        "arcana": _count_field(rows, "arcana"),
        "suit": _count_field(rows, "suit"),
        "orientation": _count_field(rows, "orientation"),
        "vi_provenance": _count_field(rows, "vi_provenance"),
        "polarity_axis": _count_field(rows, "polarity_axis"),
        "vi_orientation_attribution": _count_field(rows, "vi_orientation_attribution"),
        "meaning_en": _char_length_stats(rows, "meaning_en"),
        "meaning_vi": _char_length_stats(rows, "meaning_vi"),
        "domain_vi_coverage": _domain_vi_coverage(rows),
    }


def _spread_stats(spreads_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(spreads_path)
    by_id: dict[str, dict[str, Any]] = {}
    cards_drawn: Counter[int] = Counter()
    difficulty: Counter[str] = Counter()
    for row in rows:
        spread_id = row.get("spread_id")
        if spread_id is not None:
            by_id[str(spread_id)] = {
                "cards_drawn": row.get("cards_drawn"),
                "difficulty": row.get("difficulty"),
                "name_en": row.get("name_en"),
                "name_vi": row.get("name_vi"),
                "num_positions": len(row.get("positions") or []),
            }
        if row.get("cards_drawn") is not None:
            cards_drawn[row["cards_drawn"]] += 1
        if row.get("difficulty") is not None:
            difficulty[str(row["difficulty"])] += 1
    return {
        "count": len(rows),
        "by_spread_id": {key: by_id[key] for key in sorted(by_id)},
        "cards_drawn": {str(key): cards_drawn[key] for key in sorted(cards_drawn)},
        "difficulty": {key: difficulty[key] for key in sorted(difficulty)},
    }


def _anchor_stats(anchor_path: Path) -> dict[str, Any]:
    """Per-card_id + per-orientation counts ONLY — anchor rows have neither
    ``task_type`` nor ``cards_used``, so nothing else is derived for them."""
    rows = _read_jsonl(anchor_path)
    by_card: Counter[int] = Counter()
    by_orientation: Counter[str] = Counter()
    for row in rows:
        if row.get("card_id") is not None:
            by_card[row["card_id"]] += 1
        if row.get("orientation") is not None:
            by_orientation[str(row["orientation"])] += 1
    return {
        "count": len(rows),
        "by_card_id": {str(key): by_card[key] for key in sorted(by_card)},
        "by_orientation": {key: by_orientation[key] for key in sorted(by_orientation)},
    }
