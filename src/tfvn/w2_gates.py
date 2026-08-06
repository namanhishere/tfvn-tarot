"""Deterministic gates for the W2.2 reversed-meaning synthesis (plan W2.2).

The plan gates W2.2 on four checks. Three are mechanical and live here; the
keyword back-check (G3) is an LLM-rubric call assembled in ``w2_prompts.py``
and invoked by the build script — this module provides its deterministic
helpers (keyword containment, card-name-inline).

G1  Vietnamese-ness profile check   — function-word density + diacritic ratio
    vs ``kb/vn_register_profile.json`` (plan: pass-rate floor >= 75%).
G2  Orientation Jaccard             — token-set overlap between generated
    reversed prose and the card's authentic Vietnamese upright prose. The
    threshold is derived from the empirical distribution of authentic
    upright/reversed ``title_secondary`` pairs in the raw phatjkk source
    (plan: 90th percentile; H2).
G4  Forbidden-claims check          — deterministic blocklist for literal
    death predictions, medical diagnosis, and legal advice (plan W2.2 #4).

Also provides the W2.1 orientation-attribution proxy (polarity lexicon over
the 5 byte-identical Vietnamese fields) and the authentic-pair Jaccard
distribution used to derive the G2 threshold.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Matches the segmenter recorded in kb/vn_register_profile.json
# ("segmenter": "simple_regex_vi_tokens", version 1.0) — Vietnamese words are
# space-delimited syllables; this captures them plus any Latin tokens.
_VI_TOKEN_RE = re.compile(r"[0-9a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", re.I)

_DIACRITIC_RE = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


def simple_vi_tokens(text: str) -> List[str]:
    """Tokenise Vietnamese text with the same regex as W1.4's segmenter."""
    return _VI_TOKEN_RE.findall(text)


def token_set(text: str) -> set:
    return set(simple_vi_tokens(text.lower()))


# ------------------------------------------------------------------ W2.1 ----

# Polarity lexicon for the orientation-attribution proxy (W2.1). The 5
# byte-identical phatjkk fields are scored by the balance of positive vs
# negative tarot-domain Vietnamese terms. This is the plan's documented
# fallback ("keyword overlap as a simpler proxy") since the endpoint exposes
# no /embeddings route (verified 2026-08-06).
UPRIGHT_LEXICON = [
    "thành công", "cơ hội", "may mắn", "thuận lợi", "phát triển", "hòa hợp",
    "hạnh phúc", "thịnh vượng", "đạt được", "tích cực", "tốt đẹp", "ổn định",
    "khởi đầu", "khởi sắc", "cải thiện", "tiến triển", "viên mãn", "vui vẻ",
    "bình an", "sung túc", "đủ đầy", "gắn kết", "thấu hiểu", "chân thành",
    "nỗ lực", "kiên trì", "tự tin", "quyết tâm", "đồng thuận", "ủng hộ",
]
REVERSED_LEXICON = [
    "khó khăn", "trở ngại", "thất bại", "mất mát", "rủi ro", "lo lắng",
    "xung đột", "rạn nứt", "suy sụp", "chậm trễ", "tổn thất", "bất lợi",
    "nguy hiểm", "bế tắc", "trì trệ", "suy giảm", "khủng hoảng", "căng thẳng",
    "mâu thuẫn", "phản bội", "lừa dối", "thiếu sót", "sa sút", "đổ vỡ",
    "chán nản", "mệt mỏi", "dao động", "hoang mang", "bất an", "cô lập",
]

_UPRIGHT_RE = [re.compile(rf"\b{re.escape(w)}\b") for w in UPRIGHT_LEXICON]
_REVERSED_RE = [re.compile(rf"\b{re.escape(w)}\b") for w in REVERSED_LEXICON]


def polarity_lexicon_attribution(text: str, epsilon: float = 0.15) -> Dict[str, Any]:
    """Score the orientation-ness of Vietnamese prose via the polarity lexicon.

    Returns a dict with ``score`` in [-1, 1] (positive = upright-skew) and an
    ``attribution`` of one of: ``vi_upright`` (score >= +epsilon),
    ``vi_reversed_skew`` (score <= -epsilon), ``vi_orientation_agnostic``
    (|score| < epsilon, or no lexicon hits at all).
    """
    tokens = simple_vi_tokens(text.lower())
    if not tokens:
        return {"score": 0.0, "hits": 0, "attribution": "vi_orientation_agnostic", "epsilon": epsilon}
    up = sum(1 for r in _UPRIGHT_RE if r.search(text.lower()))
    rev = sum(1 for r in _REVERSED_RE if r.search(text.lower()))
    hits = up + rev
    score = (up - rev) / hits if hits else 0.0
    if hits == 0 or abs(score) < epsilon:
        attribution = "vi_orientation_agnostic"
    elif score > 0:
        attribution = "vi_upright"
    else:
        attribution = "vi_reversed_skew"
    return {
        "score": round(score, 4),
        "hits": hits,
        "upright_hits": up,
        "reversed_hits": rev,
        "epsilon": epsilon,
        "attribution": attribution,
    }


def authentic_profile_distances(
    vn_upright_rows: Sequence[Dict[str, Any]], profile: Dict[str, Any]
) -> List[float]:
    """Profile distances of authentic phatjkk prose (title_secondary + title_main
    of source-provenance rows). Used to calibrate the G1 threshold so it gates on
    the real Vietnamese register band instead of an arbitrary constant."""
    dists: List[float] = []
    for r in vn_upright_rows:
        if r.get("vi_provenance") != "source":
            continue
        for key in ("title_secondary", "title_main"):
            text = (r.get(key) or "").strip()
            if not text:
                continue
            d = profile_distance(text, profile)
            if d != float("inf"):
                dists.append(d)
    return dists


def percentile_90(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, int(0.90 * len(ordered)))
    return float(ordered[idx])


# -------------------------------------------------------------------- G1 ----

def profile_distance(text: str, profile: Dict[str, Any]) -> float:
    """Mean relative deviation of the text's function-word rates from the corpus.

    For each function word w with corpus rate p_w > 0, deviation is
    |t_w - p_w| / p_w; the distance is their mean. Higher = less like the
    published Vietnamese esoteric register.
    """
    tokens = simple_vi_tokens(text)
    if not tokens:
        return float("inf")
    total = len(tokens)
    counts = Counter(tokens)
    corpus = profile.get("corpus_profile") or {}
    fws = profile.get("function_words") or []
    devs = []
    for w in fws:
        p = corpus.get(w, 0.0)
        if p <= 0:
            continue
        t = counts.get(w, 0) / total
        devs.append(abs(t - p) / p)
    return float(sum(devs) / len(devs)) if devs else float("inf")


def diacritic_ratio(text: str) -> float:
    tokens = simple_vi_tokens(text)
    if not tokens:
        return 0.0
    flagged = sum(1 for t in tokens if _DIACRITIC_RE.search(t))
    return flagged / len(tokens)


def check_vietnamese_ness(
    text: str, profile: Dict[str, Any], *, max_distance: float = 2.5, min_diacritic: float = 0.5
) -> Dict[str, Any]:
    """G1: profile check. Pass when the function-word profile is close to the
    corpus AND the text carries real Vietnamese diacritics."""
    distance = profile_distance(text, profile)
    diac = diacritic_ratio(text)
    ok = distance <= max_distance and diac >= min_diacritic
    return {
        "pass": bool(ok),
        "profile_distance": round(distance, 4),
        "diacritic_ratio": round(diac, 4),
        "max_distance": max_distance,
        "min_diacritic": min_diacritic,
    }


# -------------------------------------------------------------------- G2 ----

def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard over Vietnamese-tokenised text."""
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def parse_phatjkk_records(data_txt_path: Path) -> List[Dict[str, Any]]:
    """Parse raw phatjkk data.txt into records keyed with orientation.

    The raw source stores 2 records per card (156 total): the reversed record
    has "ngược" in its ``name`` field (verified 2026-08-06). Records expose
    ``card_id`` (the "id" string), ``orientation`` (upright/reversed) and the
    six ``title_*`` fields.
    """
    text = data_txt_path.read_text(encoding="utf-8")
    # One JSON object per line in this file.
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = rec.get("name", "")
        orientation = "reversed" if "ngược" in name else "upright"
        out.append({**rec, "orientation": orientation})
    return out


def authentic_pair_jaccard_distribution(
    data_txt_path: Path, percentile: float = 0.90
) -> Dict[str, Any]:
    """Distribution of Jaccard(title_secondary_upright, title_secondary_reversed)
    over the raw phatjkk source — the plan's source for the G2 threshold.

    Returns the per-card pairs, the 90th-percentile threshold, and the count.
    If the source is missing, returns ``threshold=None`` (caller falls back).
    """
    pairs: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for rec in parse_phatjkk_records(data_txt_path):
        by_id.setdefault(rec.get("id"), {})[rec["orientation"]] = rec
    for cid, both in by_id.items():
        up = both.get("upright")
        rev = both.get("reversed")
        if not up or not rev:
            continue
        j = jaccard(up.get("title_secondary", ""), rev.get("title_secondary", ""))
        pairs.append({"card_id": cid, "jaccard": round(j, 4)})
    if not pairs:
        return {"pairs": [], "threshold": None, "count": 0, "percentile": percentile}
    values = sorted(p["jaccard"] for p in pairs)
    idx = min(len(values) - 1, int(percentile * len(values)))
    return {
        "pairs": pairs,
        "threshold": round(values[idx], 4),
        "max": round(values[-1], 4),
        "count": len(pairs),
        "percentile": percentile,
    }


# -------------------------------------------------------------------- G4 ----

_FORBIDDEN_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("death_prediction", re.compile(r"\b(sẽ\s+chết|chết\s+chóc|tử\s+vong|chết\s+người|mất\s+mạng)\b", re.I)),
    ("medical_diagnosis", re.compile(r"\b(chẩn\s+đoán|mắc\s+bệnh\s+(ung\s+thư|ung\s+thư|nan\s+y)|ung\s+thư\s+giai\s+đoạn|phát\s+hiện\s+bệnh)\b", re.I)),
    ("prescription", re.compile(r"\b(uống\s+thuốc\s+(này|đó)|liều\s+thuốc|ngừng\s+thuốc|kê\s+đơn)\b", re.I)),
    ("legal_advice", re.compile(r"\b(tư\s+vấn\s+pháp\s+lý|luật\s+sư|khởi\s+kiện\s+chắc|chắc\s+thắng\s+kiện|kiện\s+sẽ\s+thắng)\b", re.I)),
    ("financial_guarantee", re.compile(r"\b(đảm\s+bảo\s+lợi\s+nhuận|chắc\s+chắn\s+sinh\s+lời|đầu\s+tư\s+chắc\s+thắng)\b", re.I)),
]


def forbidden_claims(text: str) -> List[str]:
    """Return the list of forbidden-claim categories matched in ``text``."""
    return [name for name, pat in _FORBIDDEN_PATTERNS if pat.search(text)]


# -------------------------------------------------------- keyword helpers ----

def keyword_containment(prose: str, keywords_vi: Sequence[str]) -> float:
    """Fraction of declared Vietnamese keywords that appear verbatim in prose."""
    if not keywords_vi:
        return 0.0
    low = prose.lower()
    hits = sum(1 for k in keywords_vi if k and k.lower() in low)
    return hits / len(keywords_vi)


def card_name_inline(prose: str, name_en: str) -> bool:
    """True when the English card name appears inline in the Vietnamese prose."""
    return bool(re.search(rf"\b{re.escape(name_en)}\b", prose, re.I))
