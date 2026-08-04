"""Deterministic validators for card-name containment, orientation, keywords, aliases, Mathers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .aliases import (
    ALIAS_TABLE,
    CANONICAL_NAMES,
    NAME_TO_ID,
    assert_alias_table_total_injective,
    build_alias_table,
    resolve_name,
)

# Words that often co-occur with upright polarity in EN keywords
UPRIGHT_POLARITY_CUES = re.compile(
    r"\b(new beginnings?|success|victory|harmony|abundance|fulfilment|growth|"
    r"confidence|strength|clarity|opportunity|progress)\b",
    re.I,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_whitelist(path: Optional[Path] = None) -> dict:
    path = path or (_project_root() / "kb/card_name_whitelist.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _all_name_patterns(whitelist: Optional[dict] = None) -> List[tuple]:
    """Return list of (compiled_regex, canonical_name) sorted by name length desc."""
    wl = whitelist or {
        "canonical_names": CANONICAL_NAMES,
        "names_normalised": [n.lower() for n in CANONICAL_NAMES],
    }
    names = list(wl.get("canonical_names") or CANONICAL_NAMES)
    # Include common aliases with spaces
    extra = []
    for alias_norm, canon in ALIAS_TABLE.items():
        # restore simple title case for matching long aliases
        if alias_norm not in {n.lower() for n in names}:
            extra.append((alias_norm, canon))
    patterns = []
    for n in names:
        pat = re.compile(r"\b" + re.escape(n) + r"\b", re.I)
        patterns.append((pat, n, len(n)))
    for alias_norm, canon in extra:
        # match alias as word sequence
        pat = re.compile(r"\b" + re.escape(alias_norm) + r"\b", re.I)
        patterns.append((pat, canon, len(alias_norm)))
    patterns.sort(key=lambda x: -x[2])
    return [(p, canon) for p, canon, _ in patterns]


def extract_card_names(text: str, whitelist: Optional[dict] = None) -> List[str]:
    """Extract canonical card names mentioned in text (longest-first)."""
    found = []
    spans_taken: List[tuple] = []
    for pat, canon in _all_name_patterns(whitelist):
        for m in pat.finditer(text):
            span = m.span()
            if any(not (span[1] <= s or span[0] >= e) for s, e in spans_taken):
                continue
            spans_taken.append(span)
            found.append(canon)
    # Dedup preserve order
    out = []
    seen = set()
    for n in found:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def validate_card_name_containment(
    text: str,
    whitelist: Optional[dict] = None,
    *,
    required_names: Optional[Sequence[str]] = None,
) -> dict:
    """Pass if every card name in the text is in the whitelist.

    If required_names is given, also require each of those to appear.
    Unknown / non-whitelist card-like phrases are not currently detected beyond
    the whitelist set — fail = name that resolves outside whitelist (n/a) OR
    required name missing. For hallucinated non-deck names, callers may pass
    `forbidden_names`.
    """
    wl = whitelist
    if wl is None:
        try:
            wl = load_whitelist()
        except FileNotFoundError:
            wl = {"canonical_names": CANONICAL_NAMES, "names_normalised": [n.lower() for n in CANONICAL_NAMES]}

    allowed: Set[str] = set(wl.get("canonical_names") or CANONICAL_NAMES)
    # normalised aliases allowed as surface forms
    for a in wl.get("aliases") or []:
        allowed.add(a.get("canonical") or "")

    mentioned = extract_card_names(text, wl)
    bad = [n for n in mentioned if n not in allowed]
    missing_required = []
    if required_names:
        mentioned_set = set(mentioned)
        for n in required_names:
            try:
                cn = resolve_name(n)
            except KeyError:
                cn = n
            if cn not in mentioned_set:
                missing_required.append(cn)

    # Detect out-of-deck "The X" patterns that look like majors but aren't whitelisted
    faux = []
    for m in re.finditer(r"\bThe\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text):
        candidate = "The " + m.group(1)
        # skip if already a known card
        try:
            resolve_name(candidate)
            continue
        except KeyError:
            pass
        # skip common English phrases
        if candidate.lower() in {
            "the past",
            "the present",
            "the future",
            "the querent",
            "the reader",
            "the outcome",
            "the situation",
            "the question",
            "the cards",
            "the reading",
            "the universe",  # alias of World — resolve would catch; keep
        }:
            continue
        if candidate == "The Universe":
            continue
        faux.append(candidate)

    ok = not bad and not missing_required and not faux
    return {
        "ok": ok,
        "mentioned": mentioned,
        "not_in_whitelist": bad,
        "missing_required": missing_required,
        "hallucinated_like": faux,
    }


def validate_orientation_consistency(
    text: str,
    draw_spec: Sequence[dict],
    kb_rows: Optional[Sequence[dict]] = None,
) -> dict:
    """Fail if a reversed-draw card is described with its upright keywords.

    draw_spec: [{card_id, orientation, name_en?}]
    kb_rows: spine rows with keyword_atoms_en / polarity_axis
    """
    kb_index: Dict[tuple, dict] = {}
    if kb_rows:
        for r in kb_rows:
            kb_index[(int(r["card_id"]), r["orientation"])] = r

    violations = []
    text_l = text.lower()
    for d in draw_spec:
        cid = int(d["card_id"])
        orient = d["orientation"]
        if orient != "reversed":
            continue
        upright = kb_index.get((cid, "upright"))
        if not upright:
            # Fall back: if text asserts "xuôi" / "upright" for this card name
            name = d.get("name_en") or CANONICAL_NAMES[cid]
            if re.search(rf"{re.escape(name)}.{{0,40}}(upright|xuôi)", text, re.I):
                violations.append(
                    {"card_id": cid, "reason": "reversed_draw_described_as_upright", "name": name}
                )
            continue
        kws = upright.get("keyword_atoms_en") or upright.get("keywords_en") or []
        hits = []
        for kw in kws:
            kw = kw.strip()
            if len(kw) < 4:
                continue
            if kw.lower() in text_l:
                hits.append(kw)
        # Require multiple keyword hits to reduce false positives
        if len(hits) >= 2:
            violations.append(
                {
                    "card_id": cid,
                    "reason": "reversed_draw_has_upright_keywords",
                    "keywords": hits,
                }
            )
        # polarity null language
        name = d.get("name_en") or upright.get("name_en") or CANONICAL_NAMES[cid]
        if re.search(rf"{re.escape(name)}.{{0,60}}(upright|xuôi|bài xuôi)", text, re.I):
            violations.append(
                {"card_id": cid, "reason": "reversed_named_as_upright", "name": name}
            )

    return {"ok": len(violations) == 0, "violations": violations}


def validate_keyword_collision(
    text: str,
    draw_spec: Sequence[dict],
    kb_rows: Sequence[dict],
) -> dict:
    """Fail if distinctive keywords of an undrawn card appear in the text."""
    drawn_ids = {int(d["card_id"]) for d in draw_spec}
    text_l = text.lower()

    # Build distinctive keyword sets per card (upright + reversed keywords)
    by_card: Dict[int, Set[str]] = {}
    for r in kb_rows:
        cid = int(r["card_id"])
        kws = r.get("keyword_atoms_en") or r.get("keywords_en") or []
        by_card.setdefault(cid, set())
        for kw in kws:
            kw = kw.strip().lower()
            if len(kw) >= 5:
                by_card[cid].add(kw)

    # Distinctive = keywords that appear for at most 2 cards
    df: Dict[str, int] = {}
    for cid, kws in by_card.items():
        for kw in kws:
            df[kw] = df.get(kw, 0) + 1
    distinctive = {kw for kw, c in df.items() if c <= 2}

    collisions = []
    for cid, kws in by_card.items():
        if cid in drawn_ids:
            continue
        hits = [kw for kw in kws if kw in distinctive and kw in text_l]
        if len(hits) >= 2:
            collisions.append(
                {
                    "card_id": cid,
                    "name_en": CANONICAL_NAMES[cid],
                    "keywords": hits,
                }
            )

    return {"ok": len(collisions) == 0, "collisions": collisions}


def validate_alias_table(table: Optional[dict] = None) -> dict:
    """Pass if alias table is total over 78 canonical names and a pure function."""
    try:
        if table is None:
            assert_alias_table_total_injective()
            built = build_alias_table()
        else:
            # Validate provided table
            built = table
            for canon in CANONICAL_NAMES:
                key = " ".join(canon.strip().lower().split())
                if key not in built:
                    return {"ok": False, "error": f"missing canonical self-map for {canon}"}
                if built[key] != canon:
                    return {"ok": False, "error": f"bad self-map for {canon}"}
            # function: dict ensures one value per key
            covered = set(built.values())
            if covered != set(CANONICAL_NAMES):
                return {
                    "ok": False,
                    "error": "coverage incomplete",
                    "missing": sorted(set(CANONICAL_NAMES) - covered),
                }
        # Injectivity of the alias→canonical function is satisfied by dict.
        # Additional check: no alias key maps to non-canonical.
        for k, v in built.items():
            if v not in NAME_TO_ID:
                return {"ok": False, "error": f"alias {k!r} maps to non-canonical {v!r}"}
        return {"ok": True, "n_aliases": len(built), "n_canonical": 78}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def validate_mathers_join_guard(record: dict) -> dict:
    """Schema guard: Mathers-sourced rows must not carry numeric join keys.

    A numeric join is a type error — presence of banned keys fails the guard.
    """
    banned = {"id", "card_id", "numeric_id", "mathers_id", "number", "n", "index"}
    present = sorted(banned & set(record.keys()))
    # Also fail if someone stuffed a join key under nested "join"
    if "join" in record and isinstance(record["join"], dict):
        if any(k in record["join"] for k in ("id", "card_id", "numeric_id")):
            present.append("join.numeric")
    required = {"name_source", "meaning_upright", "meaning_reversed", "source"}
    missing_req = sorted(required - set(record.keys()))
    ok = not present and not missing_req
    return {
        "ok": ok,
        "banned_keys_present": present,
        "missing_required": missing_req,
    }


def mathers_numeric_join_is_type_error(record: dict) -> None:
    """Raise TypeError if a numeric join is attempted on a Mathers record."""
    result = validate_mathers_join_guard(record)
    if not result["ok"] and result["banned_keys_present"]:
        raise TypeError(
            f"Mathers records are not numerically joinable; banned keys present: "
            f"{result['banned_keys_present']}"
        )
