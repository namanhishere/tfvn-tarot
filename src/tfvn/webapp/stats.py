"""Statistics aggregator with mtime-keyed cache (todo A.3).

Public API of the Statistics dashboard: computes the full payload once and
caches it under a key built from the ``(mtime_ns, size)`` of every input file;
``invalidate()`` drops the cache and ``GET /api/stats`` re-computes on the
next call (e.g. after a pipeline re-run).

The payload covers:

* Combined SFT stats over ``datasets/filtered_core.jsonl`` +
  ``datasets/filtered_bulk.jsonl`` (13,571 rows), each row tagged with its
  tier.  All field access is key-presence driven — never tier driven — so a
  field like ``critique`` that exists on only a fraction of rows (2,434/5,000
  core, 4,151/8,571 bulk) is counted where it exists and bucketized as
  ``no_critique`` elsewhere.
* Split counts / split x task_type cross-tab from ``datasets/splits.json``.
* KB stats from ``kb/cards.jsonl`` and spread stats from ``kb/spreads.jsonl``.
* Anchor stats from ``datasets/anchor/anchor_readings.jsonl`` — per-card_id +
  per-orientation counts only (anchor rows carry neither ``task_type`` nor
  ``cards_used``; no fields are invented for them).

Per-section computation lives in the sibling modules ``stats_sft`` and
``stats_kb`` (250-LOC ceiling); this module owns the cache, the router and
the orchestration only.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Response

from .stats_kb import _anchor_stats, _kb_stats, _spread_stats
from .stats_sft import (
    _ifd_stats,
    _per_card_stats,
    _read_jsonl,
    _sft_distributions,
    _split_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Inputs whose (mtime_ns, size) signature keys the cache.  All are frozen,
# tracked artifacts — if any is missing, stats cannot be computed (explicit
# FileNotFoundError, never a silent partial result).
_INPUT_FILES: tuple[Path, ...] = (
    Path("datasets/filtered_core.jsonl"),
    Path("datasets/filtered_bulk.jsonl"),
    Path("datasets/splits.json"),
    Path("kb/cards.jsonl"),
    Path("kb/spreads.jsonl"),
    Path("datasets/anchor/anchor_readings.jsonl"),
)

router = APIRouter()

_lock = threading.Lock()
_cache_key: tuple[tuple[str, int, int], ...] | None = None
_cache_payload: dict[str, Any] | None = None
_compute_count = 0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _file_key(root: Path) -> tuple[tuple[str, int, int], ...]:
    """(relpath, mtime_ns, size) signature of every stats input file."""
    return tuple(
        (str(rel), (root / rel).stat().st_mtime_ns, (root / rel).stat().st_size)
        for rel in _INPUT_FILES
    )


def invalidate() -> None:
    """Drop the cached payload so the next call re-computes from disk."""
    global _cache_key, _cache_payload
    with _lock:
        _cache_key = None
        _cache_payload = None


def computation_count() -> int:
    """Number of full payload computations performed (cache misses)."""
    return _compute_count


def compute_stats(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return the stats payload, computing it once per input-file signature."""
    global _cache_key, _cache_payload, _compute_count
    key = _file_key(root)
    with _lock:
        if key == _cache_key and _cache_payload is not None:
            return _cache_payload
    payload = _compute(root)
    with _lock:
        if key != _cache_key:
            _cache_key = key
            _cache_payload = payload
            _compute_count += 1
    return payload


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/api/stats")
def get_stats() -> Response:
    """Single JSON payload with stable (sorted) key order at every level."""
    payload = compute_stats()
    return Response(
        content=json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _compute(root: Path) -> dict[str, Any]:
    core = _read_rows(root / "datasets/filtered_core.jsonl", "core")
    bulk = _read_rows(root / "datasets/filtered_bulk.jsonl", "bulk")
    rows = core + bulk
    per_card, total_reversed_percent = _per_card_stats(rows)
    return {
        "source": {
            "core_rows": len(core),
            "bulk_rows": len(bulk),
            "total": len(rows),
        },
        "tier_counts": {"core": len(core), "bulk": len(bulk)},
        "distributions": _sft_distributions(rows),
        "per_card": per_card,
        "total_reversed_percent": total_reversed_percent,
        "ifd": _ifd_stats(rows),
        "splits": _split_stats(rows, root / "datasets/splits.json"),
        "kb": _kb_stats(root / "kb/cards.jsonl"),
        "spreads": _spread_stats(root / "kb/spreads.jsonl"),
        "anchor": _anchor_stats(root / "datasets/anchor/anchor_readings.jsonl"),
    }


def _read_rows(path: Path, tier: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    for row in rows:
        row["_tier"] = tier
    return rows
