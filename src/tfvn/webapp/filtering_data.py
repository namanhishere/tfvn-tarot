"""Dataset registry + row matching for the filtering module (todo A.4).

Data layer shared by the rows and export endpoints: the dataset table
(id → path / primary-key kind / search kind), per-line iteration (missing
gitignored raw files iterate empty, never crash), and the key-presence
driven filter + substring-search predicates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
# id -> (relpath, primary-key kind, search kind).  ``all_sft`` is the union of
# the two filtered tiers, each row tagged with its ``tier``.  Missing raw files
# are tolerated (empty iteration, no crash).
# pk kinds: "example_id" | "card_orientation" | "card_id" | "spread_id" | "anchor_id"
# search kinds: "sft" | "cards" | "fallback"
DATASETS: dict[str, tuple[Optional[str], str, str]] = {
    "filtered_core": ("datasets/filtered_core.jsonl", "example_id", "sft"),
    "filtered_bulk": ("datasets/filtered_bulk.jsonl", "example_id", "sft"),
    "all_sft": (None, "example_id", "sft"),
    "anchor": ("datasets/anchor/anchor_readings.jsonl", "anchor_id", "fallback"),
    "cards": ("kb/cards.jsonl", "card_orientation", "cards"),
    "vn_spine": ("kb/vn_spine.jsonl", "card_orientation", "fallback"),
    "english_spine": ("kb/english_spine.jsonl", "card_orientation", "fallback"),
    "vn_upright": ("kb/vn_upright.jsonl", "card_id", "fallback"),
    "compact_cards": ("kb/compact_cards.jsonl", "card_orientation", "fallback"),
    "spreads": ("kb/spreads.jsonl", "spread_id", "fallback"),
    "raw_generated": ("datasets/raw/generated.jsonl", "example_id", "fallback"),
    "raw_generated_sep": ("datasets/raw/generated_sep.jsonl", "example_id", "fallback"),
    "raw_ifd_scores": ("datasets/raw/ifd_scores.jsonl", "example_id", "fallback"),
}

# (relpath, tier) per all_sft union member, in load order.
_SFT_TIERS: tuple[tuple[str, str], ...] = (
    ("datasets/filtered_core.jsonl", "core"),
    ("datasets/filtered_bulk.jsonl", "bulk"),
)


# --------------------------------------------------------------------------- #
# Row matching
# --------------------------------------------------------------------------- #

def _list_contains(value: Any, items: Any) -> bool:
    """Membership test tolerant of int/str type drift within a small list."""
    if not isinstance(items, list):
        return False
    return any(
        item == value or str(item) == str(value)
        for item in items
        if not isinstance(item, (dict, list))
    )


def card_id_matches(row: dict[str, Any], card_id: int) -> bool:
    """``card_ids`` list membership (SFT) or scalar ``card_id`` equality (KB)."""
    ids = row.get("card_ids")
    if isinstance(ids, list):
        return _list_contains(card_id, ids)
    if "card_id" in row:
        return row["card_id"] == card_id
    return False


def orientation_matches(row: dict[str, Any], orientation: str) -> bool:
    """``orientations`` list membership (SFT) or scalar ``orientation`` (KB)."""
    orientations = row.get("orientations")
    if isinstance(orientations, list):
        return orientation in orientations
    if "orientation" in row:
        return row["orientation"] == orientation
    return False


def _q_haystacks(row: dict[str, Any], search_kind: str) -> Iterator[str]:
    """Searchable text per row; ``fallback`` = any string (or str-list) value."""
    if search_kind == "sft":
        keys = ("question_vi", "reading_vi", "position_glosses")
    elif search_kind == "cards":
        keys = ("name_en", "meaning_en", "meaning_vi")
    else:
        keys = None
    if keys is not None:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        yield item
        return
    for value in row.values():
        if isinstance(value, str):
            yield value
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            yield from value


def q_matches(row: dict[str, Any], search_kind: str, q: str) -> bool:
    """Case-insensitive substring over any of the row's searchable fields."""
    needle = q.lower()
    return any(needle in hay.lower() for hay in _q_haystacks(row, search_kind))


def row_matches(
    row: dict[str, Any],
    search_kind: str,
    row_tier: Optional[str],
    q: Optional[str] = None,
    task_type: Optional[str] = None,
    register: Optional[str] = None,
    length_band: Optional[str] = None,
    querent_context: Optional[str] = None,
    spread_id: Optional[str] = None,
    card_id: Optional[int] = None,
    orientation: Optional[str] = None,
    tier: Optional[str] = None,
    ifd_min: Optional[float] = None,
    ifd_max: Optional[float] = None,
) -> bool:
    """AND semantics; every filter is optional and key-presence driven."""
    if q is not None and not q_matches(row, search_kind, q):
        return False
    if task_type is not None and row.get("task_type") != task_type:
        return False
    if register is not None and row.get("register") != register:
        return False
    if length_band is not None and row.get("length_band") != length_band:
        return False
    if querent_context is not None and row.get("querent_context") != querent_context:
        return False
    if spread_id is not None and row.get("spread_id") != spread_id:
        return False
    if card_id is not None and not card_id_matches(row, card_id):
        return False
    if orientation is not None and not orientation_matches(row, orientation):
        return False
    if row_tier is not None and tier is not None and row_tier != tier:
        return False
    if ifd_min is not None or ifd_max is not None:
        score = row.get("ifd_score")
        if score is not None:
            if ifd_min is not None and score < ifd_min:
                return False
            if ifd_max is not None and score > ifd_max:
                return False
    return True


# --------------------------------------------------------------------------- #
# Row iteration
# --------------------------------------------------------------------------- #

def iter_rows(root: Path, dataset_id: str) -> Iterator[tuple[dict[str, Any], Optional[str]]]:
    """Yield (row, tier) for every row of a dataset; missing files iterate empty."""
    if dataset_id == "all_sft":
        for rel, tier in _SFT_TIERS:
            path = root / rel
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield {**json.loads(line), "tier": tier}, tier
        return
    relpath, _pk, _search = DATASETS[dataset_id]
    if relpath is None:
        return
    path = root / relpath
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line), None


def iter_matching(root: Path, dataset_id: str, **filters: Any) -> Iterator[dict[str, Any]]:
    """Every row passing all provided filters (tier tagged for all_sft)."""
    search_kind = DATASETS[dataset_id][2]
    for row, row_tier in iter_rows(root, dataset_id):
        if row_matches(row, search_kind=search_kind, row_tier=row_tier, **filters):
            yield row
