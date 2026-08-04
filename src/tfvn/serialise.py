"""Byte-stable canonical serialiser and compact card / whitelist builders."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .aliases import ALIAS_TABLE, CANONICAL_NAMES, alias_table_for_export


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialise to UTF-8 JSON with sorted keys, compact separators, no NaN."""
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def dumps_canonical(obj: Any) -> str:
    return canonical_json_bytes(obj).decode("utf-8")


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    """Write JSONL with one canonical JSON object per line + trailing newline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [dumps_canonical(row) for row in rows]
    # Final newline for POSIX text files
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def serialise_spine_document(rows: Sequence[dict]) -> bytes:
    """Byte-stable document form of the spine (array, sorted keys per row)."""
    # Normalise each row with sorted keys via dumps round-trip, then array
    normalised = [json.loads(dumps_canonical(r)) for r in rows]
    # Stable row order: card_id, then orientation (reversed after upright)
    orient_order = {"upright": 0, "reversed": 1}
    normalised.sort(key=lambda r: (int(r["card_id"]), orient_order.get(r["orientation"], 9)))
    return canonical_json_bytes(normalised)


def build_compact_cards(spine_rows: Sequence[dict]) -> List[dict]:
    """~58-token compact form for prompt assembly."""
    out = []
    for r in spine_rows:
        kws = r.get("keyword_atoms_en") or []
        # Keep first 5 keywords for budget
        kws5 = kws[:5]
        meaning = (r.get("meaning_summary_en") or "").strip()
        # Drop parenthetical source cues for compact form
        meaning = re.sub(r"\s*\((?:Waite|Mathers):.*?\)\s*$", "", meaning)
        if len(meaning) > 180:
            meaning = meaning[:177].rstrip() + "..."
        out.append(
            {
                "card_id": int(r["card_id"]),
                "name_en": r["name_en"],
                "orientation": r["orientation"],
                "keywords_en": kws5,
                "meaning_summary_en": meaning,
                "polarity_axis": r.get("polarity_axis"),
            }
        )
    orient_order = {"upright": 0, "reversed": 1}
    out.sort(key=lambda r: (r["card_id"], orient_order.get(r["orientation"], 9)))
    return out


def compact_row_token_estimate(row: dict) -> int:
    """Approx token count without a model tokenizer (~4 chars/token EN heuristic)."""
    text = dumps_canonical(row)
    # Prefer whitespace+punct split as a closer proxy than pure chars
    parts = re.findall(r"\w+|[^\w\s]", text)
    return max(1, len(parts))


def mean_compact_tokens(rows: Sequence[dict], tokenizer=None) -> float:
    if not rows:
        return 0.0
    if tokenizer is not None:
        total = 0
        for r in rows:
            text = (
                f"{r['name_en']} {r['orientation']} "
                + " ".join(r.get("keywords_en") or [])
                + " "
                + (r.get("meaning_summary_en") or "")
            )
            total += len(tokenizer.encode(text, add_special_tokens=False))
        return total / len(rows)
    return sum(compact_row_token_estimate(r) for r in rows) / len(rows)


def build_card_name_whitelist() -> dict:
    """Single source of truth: 78 canonical names + alias table entries."""
    aliases = alias_table_for_export()
    canonical = [{"name": n, "card_id": i, "kind": "canonical"} for i, n in enumerate(CANONICAL_NAMES)]
    alias_entries = [
        {
            "name": a["alias"],
            "canonical": a["canonical"],
            "card_id": a["card_id"],
            "kind": "alias",
        }
        for a in aliases
        # skip pure self-maps already in canonical list as canonical kind
        if a["alias"] != a["canonical"].lower()
        and a["alias"] != " ".join(a["canonical"].lower().split())
    ]
    # Also include surface forms (original casing) of canonical names lowercased
    names_lower = sorted({n.lower() for n in CANONICAL_NAMES} | {a["alias"] for a in aliases})
    return {
        "canonical_names": CANONICAL_NAMES,
        "canonical_count": 78,
        "aliases": aliases,
        "names_normalised": names_lower,
        "entries": canonical + alias_entries,
        "entry_count": 78 + len(alias_entries),
    }


def try_load_qwen_tokenizer():
    """Load Qwen tokenizer if available; return None on failure."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        return tok
    except Exception:
        return None
