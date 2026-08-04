"""Build the 156-row English semantic spine (78 cards × 2 orientations).

Sources joined on the canonical numeric spine 0–77:
  - tarotoo_cards.json (keywords, modern meanings)
  - StarTarotOnline historical Waite divinatory meanings
  - Blacik / lindseyb / smallcat419 (cross-check only)
  - Mathers 1888 (name-join only; never numeric) for Two of Cups reversed
  - Raw Waite PKT for Reversed; semicolon provenance check
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .aliases import CANONICAL_NAMES, NAME_TO_ID, resolve_name
from .serialise import dumps_canonical, write_jsonl

# Mathers rows must NEVER carry a numeric join key (schema guard).
MATHERS_ALLOWED_KEYS = frozenset(
    {"name_source", "name_canonical", "meaning_upright", "meaning_reversed", "source"}
)

REV_SOURCES = frozenset({"waite", "mathers", "derived"})

# Polarity for reversed: keep 2–3 separable categories.
POLARITY_BY_KEYWORD: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(delay|delays|waiting|stagnation|postpon)", re.I), "delayed"),
    (re.compile(r"\b(block|blocked|obstacle|opposition|stoppage|check|hindrance)", re.I), "blocked"),
    (re.compile(r"\b(excess|over-|too much|abuse|extreme)", re.I), "excess"),
    (re.compile(r"\b(internal|inward|withdrawal|isolation|secret)", re.I), "internalized"),
    (re.compile(r"\b(opposite|invert|reversal|contrary|antithes)", re.I), "inverted"),
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_mathers_1888(html_path: Optional[Path] = None) -> List[dict]:
    """Parse Mathers card meanings. Returns name-keyed records with NO numeric id."""
    path = html_path or (_project_root() / "data/pd-texts/mathers_1888_03.html")
    html = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text)

    # Split into numbered entries 1..N (and 0 if present). Mathers minors are
    # numbered 22..77 with King→Ace descending — we ignore those numbers.
    parts = re.split(r"(?=(?<!\d)\d{1,2}\s*\.\s+(?:The\s+|[A-Z]))", text)
    by_canon: Dict[str, dict] = {}

    entry_re = re.compile(
        r"^(?P<n>\d{1,2})\s*\.\s+"
        r"(?P<body>.+)$",
        re.S,
    )
    # Name ends at `. --` or `.--` or `.` followed by capital meaning, then
    # upright until `; R.` or `: R.`, then reversed until period before next card.
    name_mean_re = re.compile(
        r"^(?P<name>.+?)\s*"
        r"(?:\.\s*-{1,2}\s*|\.\s+(?=[A-Z])|-+\s*)"
        r"(?P<up>.+?)\s*[;:]\s*R\.\s*"
        r"(?P<rev>.+?)(?:\s*\.\s*)?$",
        re.S,
    )

    for part in parts:
        part = part.strip()
        if not part:
            continue
        em = entry_re.match(part)
        if not em:
            continue
        body = em.group("body").strip()
        # Truncate at next card number if split leaked
        body = re.split(r"(?=\s\d{1,2}\s*\.\s+(?:The\s+|[A-Z]))", body)[0].strip()
        mm = name_mean_re.match(body)
        if not mm:
            continue
        raw_name = mm.group("name").strip().rstrip(".").strip()
        # Clean trailing "or Pope" etc. left intact for alias table
        up = re.sub(r"\s+", " ", mm.group("up")).strip(" .;")
        rev = re.sub(r"\s+", " ", mm.group("rev")).strip(" .;")
        # Drop trailing next-card fragments
        rev = re.split(r"\s+\d{1,2}\s*\.\s+", rev)[0].strip()
        try:
            canon = resolve_name(raw_name)
        except KeyError:
            base = re.split(r",\s*or\s+", raw_name)[0].strip()
            base = re.split(r"\s+or\s+", base)[0].strip()
            try:
                canon = resolve_name(base)
            except KeyError:
                continue
        if canon in by_canon:
            continue
        out = {
            "name_source": raw_name,
            "name_canonical": canon,
            "meaning_upright": up,
            "meaning_reversed": rev,
            "source": "mathers_1888",
        }
        _assert_mathers_schema(out)
        by_canon[canon] = out

    # Manual exceptions for stubborn cards
    manual = {
        "The Hierophant or Pope": (
            "Mercy, Beneficence Kindness, Goodness",
            "Over-kindness, weakness, Foolish exercise of generosity",
        ),
        "Themis, or Justice": (
            "Equilibrium, Balance, Justice",
            "Bigotry, Want of Balance, Abuse of Justice, Over-severity, Inequality, Bias",
        ),
        "Five of Pentacles": (
            "Lover or Mistress, Love, Sweetness, Affection, Pure and Chaste Love",
            "Disgraceful Love, Imprudence, License, Profligacy",
        ),
        "Deuce of Cups": (
            "Love, Attachment, Friendship, Sincerity, Affection",
            "Crossed desires, Obstacles, Opposition, Hindrance",
        ),
    }
    for raw_name, (up, rev) in manual.items():
        try:
            canon = resolve_name(raw_name)
        except KeyError:
            canon = resolve_name(re.split(r",\s*or\s+| or ", raw_name)[0].strip())
        if canon in by_canon:
            continue
        out = {
            "name_source": raw_name,
            "name_canonical": canon,
            "meaning_upright": up,
            "meaning_reversed": rev,
            "source": "mathers_1888",
        }
        _assert_mathers_schema(out)
        by_canon[canon] = out

    return list(by_canon.values())


def _assert_mathers_schema(rec: dict) -> None:
    extra = set(rec.keys()) - MATHERS_ALLOWED_KEYS - {"name_canonical"}
    # name_canonical is allowed for our internal use; strip before any external
    # numeric join surface. Forbidden keys:
    for banned in ("id", "card_id", "numeric_id", "mathers_id", "number", "n"):
        if banned in rec:
            raise AssertionError(f"Mathers record must not carry numeric join key {banned!r}")


def mathers_by_canonical(records: Optional[List[dict]] = None) -> Dict[str, dict]:
    recs = records if records is not None else parse_mathers_1888()
    return {r["name_canonical"]: r for r in recs}


def waite_has_semicolon_reversed(waite_txt_path: Optional[Path] = None) -> bool:
    """Sanity: Four of Pentacles uses Reversed; in the raw PKT text."""
    path = waite_txt_path or (_project_root() / "data/pd-texts/waite_pictorial_key_1911.txt")
    text = path.read_text(encoding="utf-8")
    # Four of Pentacles block
    m = re.search(
        r"surety of possessions.*?Reversed[;:].*?Suspense,\s*delay,\s*opposition",
        text,
        re.S | re.I,
    )
    if not m:
        return False
    return "Reversed;" in m.group(0)


def two_of_cups_waite_missing_reversed(star_row: dict) -> bool:
    """True when Waite §2 has no reversed line (Star notes §4-only or empty)."""
    dm = star_row.get("divinatory_meanings") or {}
    section = (dm.get("reversed_section") or "").lower()
    if "no reversed" in section or "§4" in (dm.get("reversed_section") or "") or "§4" in section:
        return True
    # Also treat single-token fragment from additional meanings as insufficient
    rev = (dm.get("reversed") or "").strip()
    if rev.lower() in {"", "passion.", "passion"}:
        return True
    return False


def _keywords_from_sources(
    tarotoo: dict,
    lindsey: Optional[dict],
    blacik: Optional[dict],
    orientation: str,
) -> List[str]:
    atoms: List[str] = []
    if orientation == "upright":
        atoms.extend(tarotoo.get("keywords_upright") or [])
        if lindsey:
            atoms.extend((lindsey.get("meanings") or {}).get("upright") or [])
        if blacik:
            raw = blacik.get("upright_meaning") or ""
            atoms.extend([p.strip() for p in raw.split(",") if p.strip()])
    else:
        atoms.extend(tarotoo.get("keywords_reversed") or [])
        if lindsey:
            atoms.extend((lindsey.get("meanings") or {}).get("reversed") or [])
        if blacik:
            raw = blacik.get("reversed_meaning") or ""
            atoms.extend([p.strip() for p in raw.split(",") if p.strip()])
    # Dedup case-insensitively, preserve order, cap 9
    seen = set()
    out: List[str] = []
    for a in atoms:
        a = re.sub(r"\s+", " ", a.strip())
        if not a:
            continue
        k = a.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
        if len(out) >= 9:
            break
    # Ensure at least 5 by splitting meaning if needed
    if len(out) < 5:
        meaning = (
            tarotoo.get("meaning_upright")
            if orientation == "upright"
            else tarotoo.get("meaning_reversed")
        ) or ""
        for part in re.split(r"[,;.]", meaning):
            part = part.strip()
            if len(part) < 3:
                continue
            k = part.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(part)
            if len(out) >= 5:
                break
    return out[:9]


def _polarity(orientation: str, keywords: List[str], meaning: str) -> Optional[str]:
    if orientation == "upright":
        return None
    blob = " ".join(keywords) + " " + meaning
    for pat, label in POLARITY_BY_KEYWORD:
        if pat.search(blob):
            return label
    return "inverted"


def _summarise_meaning(
    orientation: str,
    tarotoo: dict,
    star: dict,
    mathers_rev: Optional[str],
    use_mathers_rev: bool,
) -> Tuple[str, str]:
    """Return (meaning_summary_en, reversed_provenance for reversed rows only)."""
    dm = star.get("divinatory_meanings") or {}
    if orientation == "upright":
        modern = (tarotoo.get("meaning_upright") or "").strip()
        historical = (dm.get("upright") or "").strip()
        # Distilled: prefer modern keywords-as-prose + short historical cue
        summary = modern
        if historical and historical.lower() not in modern.lower():
            # keep summary curated, not verbatim long Waite — truncate historical cue
            cue = historical if len(historical) < 160 else historical[:157] + "..."
            summary = f"{modern.rstrip('.')} (Waite: {cue})"
        return summary, "waite"

    # reversed
    if use_mathers_rev and mathers_rev:
        modern = (tarotoo.get("meaning_reversed") or "").strip()
        summary = f"{modern.rstrip('.')} (Mathers: {mathers_rev})"
        return summary, "mathers"

    modern = (tarotoo.get("meaning_reversed") or "").strip()
    historical = (dm.get("reversed") or "").strip()
    summary = modern
    if historical and historical.lower() not in modern.lower():
        cue = historical if len(historical) < 160 else historical[:157] + "..."
        summary = f"{modern.rstrip('.')} (Waite: {cue})"
    return summary, "waite"


def build_english_spine(data_root: Optional[Path] = None) -> List[dict]:
    root = data_root or (_project_root() / "data")
    tarotoo = _load_json(root / "github/tarotoo_cards.json")
    tarotoo_by_id = {int(c["id"]): c for c in tarotoo}

    star_rows = []
    with open(root / "hf/StarTarotOnline__tarot-rws-historical-meanings/cards.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                star_rows.append(json.loads(line))
    star_by_id = {int(r["numeric_id"]): r for r in star_rows}

    blacik = _load_json(root / "hf/Blacik__deckaura-tarot-card-meanings/tarot_card_meanings.json")
    blacik_by_name = {resolve_name(c["card_name"]): c for c in blacik["cards"]}

    lindsey = _load_json(root / "github/lindseyb_tarot.json")
    lindsey_by_name = {resolve_name(c["name"]): c for c in lindsey["cards"]}

    smallcat = _load_json(root / "github/smallcat419_index.json")
    smallcat_by_id = {int(c["id"]): c for c in smallcat["cards"]}

    mathers = mathers_by_canonical()

    # Verify five-source numeric spine agreement (0 conflicts on names)
    for cid in range(78):
        t_name = tarotoo_by_id[cid]["name"]
        s_name = star_by_id[cid]["name"]
        sc_name = smallcat_by_id[cid]["name"]
        assert resolve_name(t_name) == resolve_name(s_name) == resolve_name(sc_name) == CANONICAL_NAMES[cid], (
            cid,
            t_name,
            s_name,
            sc_name,
        )

    assert waite_has_semicolon_reversed(root / "pd-texts/waite_pictorial_key_1911.txt")

    rows: List[dict] = []
    for cid in range(78):
        name = CANONICAL_NAMES[cid]
        t = tarotoo_by_id[cid]
        s = star_by_id[cid]
        b = blacik_by_name.get(name)
        l = lindsey_by_name.get(name)
        m = mathers.get(name)

        # Two of Cups: Waite §2 has no reversed line — Mathers is required provenance.
        use_mathers = name == "Two of Cups"
        if use_mathers and m is None:
            raise AssertionError("Mathers record required for Two of Cups reversed but missing")

        for orientation in ("upright", "reversed"):
            keywords = _keywords_from_sources(t, l, b, orientation)
            if orientation == "reversed" and use_mathers and m:
                # Prefer Mathers atoms for reversed Two of Cups
                m_atoms = [p.strip() for p in m["meaning_reversed"].split(",") if p.strip()]
                for a in m_atoms:
                    if a.lower() not in {k.lower() for k in keywords}:
                        keywords.append(a)
                keywords = keywords[:9]
                if len(keywords) < 5:
                    keywords = (keywords + m_atoms)[:9]

            mathers_rev = m["meaning_reversed"] if m else None
            meaning, prov = _summarise_meaning(
                orientation, t, s, mathers_rev, use_mathers and orientation == "reversed"
            )
            if use_mathers and orientation == "reversed":
                prov = "mathers"
            polarity = _polarity(orientation, keywords, meaning)

            row = {
                "card_id": cid,
                "name_en": name,
                "orientation": orientation,
                "arcana": t.get("arcana"),
                "suit": t.get("suit"),
                "number": t.get("number_numerology"),
                "element": t.get("element"),
                "planet": t.get("planet"),
                "zodiac": t.get("zodiac"),
                "keyword_atoms_en": keywords,
                "meaning_summary_en": meaning,
                "polarity_axis": polarity,
            }
            if orientation == "reversed":
                row["reversed_provenance"] = prov
            else:
                row["reversed_provenance"] = None
            rows.append(row)

    return rows


def assert_spine(rows: List[dict]) -> None:
    assert len(rows) == 156, f"expected 156 rows, got {len(rows)}"
    by_key = {(r["card_id"], r["orientation"]): r for r in rows}
    for cid in range(78):
        assert (cid, "upright") in by_key
        assert (cid, "reversed") in by_key

    for r in rows:
        if r["orientation"] == "reversed":
            assert r["reversed_provenance"] in REV_SOURCES
            assert r["reversed_provenance"] != "derived", f"derived not allowed: {r['name_en']}"
        assert 5 <= len(r["keyword_atoms_en"]) <= 9, (r["name_en"], r["orientation"], r["keyword_atoms_en"])

    mathers_rows = [r for r in rows if r.get("reversed_provenance") == "mathers"]
    assert len(mathers_rows) >= 1
    assert any(r["name_en"] == "Two of Cups" for r in mathers_rows), mathers_rows

    fop = by_key[(NAME_TO_ID["Four of Pentacles"], "reversed")]
    assert fop["reversed_provenance"] == "waite", fop

    # No derived
    assert sum(1 for r in rows if r.get("reversed_provenance") == "derived") == 0


def write_spine(out_path: Optional[Path] = None, data_root: Optional[Path] = None) -> Path:
    rows = build_english_spine(data_root=data_root)
    assert_spine(rows)
    path = out_path or (_project_root() / "kb/english_spine.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
    return path
