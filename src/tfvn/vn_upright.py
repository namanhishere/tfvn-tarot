"""Vietnamese upright layer: name-keyed, defect-fixed records from phatjkk data.txt."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .aliases import CANONICAL_NAMES, NAME_TO_ID, resolve_name
from .serialise import write_jsonl

DOMAIN_FIELDS = ("title_main", "title_secondary", "title_love", "title_work", "title_money", "title_health")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _fix_suc_khoe(text: str) -> str:
    # Normalise the common variant with different tone on e
    return text.replace("sức khoẻ", "sức khỏe").replace("Sức khoẻ", "Sức khỏe")


def _strip_boilerplate(field: str, text: str, card_name: str) -> str:
    """Strip leading domain headers where present.

    Covers the common ``Về …`` / ``Trong …`` headers AND the four per-record
    boilerplate exceptions (plan W1.2):

    - id 22 (The World): card name injected between ``Về`` and ``trong tình yêu``
    - id 48 (Knight of Cups): ``Trong công việc`` / ``Trong vấn đề tài chính`` /
      ``Trong khía cạnh sức khỏe`` variants
    - id 54 (Four of Swords): trailing period after ``Về tình yêu``
    - id 78 (King of Pentacles): ``Về tài chính`` (already covered generically)
    """
    t = text.strip()
    punct = r"[:：\-.]?"
    headers = [
        # Common "Về …" headers (punct optional so a trailing "." is consumed)
        rf"^Về\s+tình\s+yêu\s*{punct}\s*",
        rf"^Về\s+công\s+việc\s*{punct}\s*",
        rf"^Về\s+tiền\s+bạc\s*{punct}\s*",
        rf"^Về\s+tài\s+chính\s*{punct}\s*",
        rf"^Về\s+sức\s+khỏe\s*{punct}\s*",
        rf"^Về\s+sức\s+khoẻ\s*{punct}\s*",
        # Common "Trong …" headers
        rf"^Trong\s+tình\s+yêu\s*{punct}\s*",
        rf"^Trong\s+công\s+việc\s*{punct}\s*",
        rf"^Trong\s+vấn\s+đề\s+tài\s+chính\s*{punct}\s*",
        rf"^Trong\s+khía\s+cạnh\s+sức\s+khỏe\s*{punct}\s*",
        # Exception id 22: card name injected between "Về" and "trong tình yêu".
        # Lazy .+? stays on the first line (no re.S), so it cannot cross content.
        rf"^Về\s+.+?\s+trong\s+tình\s+yêu\s*{punct}\s*",
    ]
    for h in headers:
        t = re.sub(h, "", t, flags=re.I)
    return t.strip()


def _parse_card_name(raw_name: str) -> Tuple[str, str]:
    """Return (canonical_en, orientation) from 'Lá bài The Fool ngược' etc."""
    name = raw_name.replace("\n", " ").replace("\r", " ")
    name = re.sub(r"\s+", " ", name).strip()
    # Strip prefix
    name = re.sub(r"^Lá\s+bài\s+", "", name, flags=re.I).strip()
    orientation = "upright"
    if name.endswith(" ngược") or name.endswith("ngược"):
        orientation = "reversed"
        name = re.sub(r"\s*ngược\s*$", "", name).strip()
    # name should now be English card name
    canon = resolve_name(name)
    return canon, orientation


def load_raw_vietnamese_records(path: Optional[Path] = None) -> List[dict]:
    path = path or (_project_root() / "data/vietnamese/Tarot-Vietnamese-API/data.txt")
    raw = path.read_bytes()
    # File may lack trailing newline (155 lines / 156 records if sized by wc -l)
    text = raw.decode("utf-8")
    recs = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        recs.append(json.loads(line))
    return recs


def build_vn_upright(data_path: Optional[Path] = None) -> List[dict]:
    """Build 78 name-keyed upright Vietnamese records (+ Page placeholder)."""
    raw = load_raw_vietnamese_records(data_path)

    # Collect by (canonical, orientation)
    buckets: Dict[Tuple[str, str], List[dict]] = {}
    for rec in raw:
        # Fix id 48 embedded newline in name before parse
        name_field = rec.get("name") or ""
        if "\n" in name_field:
            rec = dict(rec)
            rec["name"] = name_field.replace("\n", " ")
        canon, orient = _parse_card_name(rec["name"])
        buckets.setdefault((canon, orient), []).append(rec)

    # Knight of Pentacles duplicate: assert byte-identical prose for upright pair
    knight_u = buckets.get(("Knight of Pentacles", "upright"), [])
    if len(knight_u) >= 2:
        # Compare domain fields
        def prose_tuple(r):
            return tuple(
                (r.get(k) or "")
                for k in (
                    "title_main",
                    "title_secondary",
                    "title_love",
                    "title_work",
                    "title_money",
                    "title_heath",
                    "title_health",
                )
            )

        base = prose_tuple(knight_u[0])
        for other in knight_u[1:]:
            if prose_tuple(other) != base:
                raise AssertionError(
                    "Knight of Pentacles upright duplicates diverged — source drift"
                )
        buckets[("Knight of Pentacles", "upright")] = [knight_u[0]]

    knight_r = buckets.get(("Knight of Pentacles", "reversed"), [])
    if len(knight_r) >= 2:
        def prose_tuple(r):
            return tuple(
                (r.get(k) or "")
                for k in (
                    "title_main",
                    "title_secondary",
                    "title_love",
                    "title_work",
                    "title_money",
                    "title_heath",
                    "title_health",
                )
            )

        base = prose_tuple(knight_r[0])
        for other in knight_r[1:]:
            if prose_tuple(other) != base:
                raise AssertionError(
                    "Knight of Pentacles reversed duplicates diverged — source drift"
                )
        buckets[("Knight of Pentacles", "reversed")] = [knight_r[0]]

    # Page of Pentacles should be missing
    assert ("Page of Pentacles", "upright") not in buckets, "Page unexpectedly present"

    out: List[dict] = []
    for cid, canon in enumerate(CANONICAL_NAMES):
        if canon == "Page of Pentacles":
            out.append(
                {
                    "card_id": cid,
                    "name_en": canon,
                    "vi_provenance": "synthetic_no_anchor",
                    "title_main": "",
                    "title_secondary": "",
                    "title_love": "",
                    "title_work": "",
                    "title_money": "",
                    "title_health": "",
                    "source_ids": [],
                }
            )
            continue

        uprights = buckets.get((canon, "upright"), [])
        if not uprights:
            raise AssertionError(f"missing upright Vietnamese for {canon}")
        rec = uprights[0]

        # Source id 13 = The Hanged Man has work/money headers swapped.
        # Also auto-detect if field body starts with the wrong domain header.
        work = rec.get("title_work") or ""
        money = rec.get("title_money") or ""
        work_has_money_hdr = bool(re.match(r"^\s*Về\s+(tiền|tài\s+chính)", work, re.I))
        money_has_work_hdr = bool(re.match(r"^\s*Về\s+công\s+việc", money, re.I))
        if canon == "The Hanged Man" or (work_has_money_hdr and money_has_work_hdr):
            work, money = money, work

        health = rec.get("title_health") or rec.get("title_heath") or ""

        fields = {
            "title_main": rec.get("title_main") or "",
            "title_secondary": rec.get("title_secondary") or "",
            "title_love": rec.get("title_love") or "",
            "title_work": work,
            "title_money": money,
            "title_health": health,
        }
        for k, v in list(fields.items()):
            v = _nfc(_fix_suc_khoe(v))
            v = v.replace("\r\n", "\n").replace("\r", "\n")
            v = _strip_boilerplate(k, v, canon)
            fields[k] = v

        out.append(
            {
                "card_id": cid,
                "name_en": canon,
                "vi_provenance": "source",
                **fields,
                "source_ids": [str(rec.get("id", ""))],
            }
        )

    return out


def assert_vn_upright(rows: List[dict]) -> None:
    assert len(rows) == 78
    by_name = {r["name_en"]: r for r in rows}
    assert len(by_name) == 78, "duplicate names"
    assert by_name["Page of Pentacles"]["vi_provenance"] == "synthetic_no_anchor"
    assert by_name["Knight of Pentacles"]["vi_provenance"] == "source"
    # Knight appears once
    assert sum(1 for r in rows if r["name_en"] == "Knight of Pentacles") == 1

    blob = json.dumps(rows, ensure_ascii=False)
    assert "title_heath" not in blob
    assert "sức khoẻ" not in blob  # normalised away

    # Names must not carry embedded newlines (id-48 defect class)
    for r in rows:
        assert "\n" not in r["name_en"]
        assert "\r" not in r["name_en"]
        for f in DOMAIN_FIELDS:
            assert "\r" not in (r.get(f) or "")

    # All provenance set
    for r in rows:
        assert r["vi_provenance"] in {"source", "synthetic_no_anchor"}

    # W1.2: four per-record boilerplate exceptions, handled (verified by enumeration)
    exceptions = {
        "The World": {"title_love": r"^Về\s+The World\s+trong\s+tình\s+yêu"},
        "Knight of Cups": {
            "title_work": r"^Trong\s+công\s+việc",
            "title_money": r"^Trong\s+vấn\s+đề\s+tài\s+chính",
            "title_health": r"^Trong\s+khía\s+cạnh\s+sức\s+khỏe",
        },
        "Four of Swords": {"title_love": r"^\."},
        "King of Pentacles": {"title_money": r"^Về\s+tài\s+chính"},
    }
    for name, field_pats in exceptions.items():
        row = by_name[name]
        for f, pat in field_pats.items():
            assert not re.match(pat, row.get(f) or "", re.I), (
                f"boilerplate exception unhandled: {name}.{f} starts with {pat!r}"
            )

    # The Hanged Man (source id 13): work/money headers were swapped in source
    hanged = by_name["The Hanged Man"]
    assert hanged["vi_provenance"] == "source"
    # After fix, work field should not still start with money header
    assert not re.match(r"^\s*Về\s+(tiền|tài\s+chính)", hanged["title_work"] or "", re.I)
    assert not re.match(r"^\s*Về\s+công\s+việc", hanged["title_money"] or "", re.I)
    # Content should discuss the right domain
    assert "công việc" in (hanged["title_work"] or "").lower() or "việc" in (hanged["title_work"] or "").lower()


def write_vn_upright(out_path: Optional[Path] = None, data_path: Optional[Path] = None) -> Path:
    rows = build_vn_upright(data_path=data_path)
    assert_vn_upright(rows)
    path = out_path or (_project_root() / "kb/vn_upright.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
    return path
