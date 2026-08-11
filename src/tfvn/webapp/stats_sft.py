"""SFT stats computation for the dataset viewer (todo A.3 helper module).

Owns the combined core+bulk union statistics: field distributions (all
key-presence driven), per-card orientation mix + frequency, IFD stats with a
10-bin histogram, and the splits join.  Consumed by ``stats.py``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

_IFD_BINS = 10


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Field distributions
# ---------------------------------------------------------------------------

def _count_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Count non-None values of *field* across rows; sorted keys, absent -> 0."""
    counter = Counter(r.get(field) for r in rows)
    non_null = {key: count for key, count in counter.items() if key is not None}
    return {str(key): non_null[key] for key in sorted(non_null)}


def _count_cards_drawn(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(len(r.get("cards_used") or []) for r in rows)
    return {str(key): counter[key] for key in sorted(counter)}


def _count_critique_verdict(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Key-presence driven: rows without a ``critique.verdict`` land in the
    ``no_critique`` bucket — never skipped and never tier-assumed."""
    counter: Counter[str] = Counter()
    for row in rows:
        critique = row.get("critique")
        if isinstance(critique, dict) and critique.get("verdict") is not None:
            counter[str(critique["verdict"])] += 1
        else:
            counter["no_critique"] += 1
    return {key: counter[key] for key in sorted(counter)}


def _count_provenance(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for tag in row.get("provenance") or []:
            if tag is not None:
                counter[str(tag)] += 1
    return {key: counter[key] for key in sorted(counter)}


def _sft_distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "task_type": _count_field(rows, "task_type"),
        "register": _count_field(rows, "register"),
        "length_band": _count_field(rows, "length_band"),
        "querent_context": _count_field(rows, "querent_context"),
        "spread_id": _count_field(rows, "spread_id"),
        "spread_name_vi": _count_field(rows, "spread_name_vi"),
        "cards_drawn": _count_cards_drawn(rows),
        "critique_applied": _count_field(rows, "critique_applied"),
        "critique_verdict": _count_critique_verdict(rows),
        "safety_category": _count_field(rows, "safety_category"),
        "grounding_defect": _count_field(rows, "grounding_defect"),
        "wrong_claim": _count_field(rows, "wrong_claim"),
        "provenance": _count_provenance(rows),
    }


# ---------------------------------------------------------------------------
# Per-card orientation mix + frequency
# ---------------------------------------------------------------------------

def _per_card_stats(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    """Orientation mix + frequency per card, and the global reversed percent."""
    frequency: Counter[int] = Counter()
    mix: dict[int, dict[str, int]] = {}
    total_entries = 0
    reversed_entries = 0
    for row in rows:
        for used in row.get("cards_used") or []:
            card_id = used.get("card_id")
            if card_id is None:
                continue
            frequency[card_id] += 1
            orientation = (
                "reversed" if used.get("orientation") == "reversed" else "upright"
            )
            cell = mix.setdefault(card_id, {"upright": 0, "reversed": 0})
            cell[orientation] += 1
            total_entries += 1
            reversed_entries += orientation == "reversed"
    reversed_percent = (
        round(100.0 * reversed_entries / total_entries, 2) if total_entries else 0.0
    )
    return {
        "frequency": {str(card_id): frequency[card_id] for card_id in sorted(frequency)},
        "orientation_mix": {
            str(card_id): mix[card_id] for card_id in sorted(mix)
        },
        "total_card_mentions": total_entries,
    }, reversed_percent


# ---------------------------------------------------------------------------
# IFD stats
# ---------------------------------------------------------------------------

def _ifd_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """min/max/mean + 10-bin histogram; missing-key tolerant (defensive)."""
    values = [
        row["ifd_score"]
        for row in rows
        if isinstance(row.get("ifd_score"), (int, float))
    ]
    empty = {
        "count": 0,
        "min": None,
        "max": None,
        "mean": None,
        "histogram": [0] * _IFD_BINS,
        "bin_edges": [],
    }
    if not values:
        return empty
    lo, hi = min(values), max(values)
    mean = sum(values) / len(values)
    if hi == lo:
        histogram = [len(values)] + [0] * (_IFD_BINS - 1)
        edges = [lo] * (_IFD_BINS + 1)
    else:
        width = (hi - lo) / _IFD_BINS
        histogram = [0] * _IFD_BINS
        for value in values:
            histogram[min(int((value - lo) / width), _IFD_BINS - 1)] += 1
        edges = [lo + i * width for i in range(_IFD_BINS + 1)]
        edges[-1] = hi
    return {
        "count": len(values),
        "min": round(lo, 6),
        "max": round(hi, 6),
        "mean": round(mean, 6),
        "histogram": histogram,
        "bin_edges": [round(edge, 6) for edge in edges],
    }


# ---------------------------------------------------------------------------
# Splits join
# ---------------------------------------------------------------------------

def _split_stats(rows: list[dict[str, Any]], splits_path: Path) -> dict[str, Any]:
    with splits_path.open(encoding="utf-8") as fh:
        splits: dict[str, str] = json.load(fh)
    counts: Counter[str] = Counter()
    cross: dict[str, Counter[str]] = {}
    unmatched = 0
    for row in rows:
        split = splits.get(row.get("example_id"))
        if split is None:
            unmatched += 1
            continue
        counts[split] += 1
        task_type = row.get("task_type")
        if task_type is not None:
            cross.setdefault(split, Counter())[str(task_type)] += 1
    return {
        "counts": {key: counts[key] for key in sorted(counts)},
        "by_task_type": {
            split: {task: counter[task] for task in sorted(counter)}
            for split, counter in sorted(cross.items())
        },
        "unmatched_rows": unmatched,
        "total_rows": len(rows),
    }
