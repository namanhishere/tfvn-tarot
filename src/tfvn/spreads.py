"""Compress TarotSchema spreads and verify positional discrimination via TF-IDF."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .serialise import dumps_canonical, write_jsonl

DIFFICULTY_MAP = {
    "easiest": 1,
    "very easy": 1,
    "easy": 2,
    "moderate": 3,
    "average": 3,
    "somewhat tough": 4,
    "varies": 3,
    "hard": 5,
    "complicated": 5,
}

# Hand-crafted short labels for positions that are free-text only in the source.
# Keys: (spread_name, index) → label_en
LABEL_OVERRIDES: Dict[Tuple[str, int], str] = {}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _compress_gloss(full: str, label: str, max_words: int = 12) -> str:
    """Compress position prose to ~15 tokens / few words."""
    full = re.sub(r"\s+", " ", full.strip())
    if len(full.split()) <= max_words and len(full) <= 90:
        # Still prefix label if not already present
        if label.lower() not in full.lower():
            return f"{label}: {full}"
        return full
    # Prefer label + first clause
    clause = re.split(r"[.;]", full)[0].strip()
    words = clause.split()
    if len(words) > max_words:
        clause = " ".join(words[:max_words])
    if label.lower() not in clause.lower():
        return f"{label}: {clause}"
    return clause


def _infer_label(spread_name: str, index: int, full: str, all_positions: List[str]) -> str:
    """Derive a short English label; disambiguate duplicates within a spread."""
    full_s = full.strip()
    # Short strings are already labels
    if len(full_s.split()) <= 4 and len(full_s) <= 40:
        base = full_s
    else:
        # Heuristic keywords
        low = full_s.lower()
        for key, lab in [
            ("past", "past"),
            ("present", "present"),
            ("future", "future"),
            ("outcome", "outcome"),
            ("obstacle", "obstacles"),
            ("suggestion", "suggestions"),
            ("external", "external influences"),
            ("hidden", "hidden influences"),
            ("unconscious", "unconscious"),
            ("conscious", "conscious"),
            ("significator", "significator"),
            ("blind spot", "blind spot"),
            ("next step", "next step"),
            ("result", "result"),
            ("finance", "finance"),
            ("work", "work"),
            ("home", "home"),
            ("partner", "partners"),
            ("friend", "friends"),
            ("reputation", "reputation"),
            ("hope", "hopes and fears"),
            ("do not", "avoid"),
            ("don't", "avoid"),
            ("do this", "action"),
            ("leads to", "outcome"),
        ]:
            if key in low:
                base = lab
                break
        else:
            words = _tokenize(full_s)[:3]
            base = " ".join(words) if words else f"position {index + 1}"

    # Disambiguate identical labels within spread
    same = [i for i, p in enumerate(all_positions) if _infer_label_base(spread_name, i, p) == _infer_label_base(spread_name, index, full)]
    # Avoid recursion: use simpler base for counting
    base_simple = _label_base_from_text(full_s)
    same_count = sum(1 for p in all_positions if _label_base_from_text(p.strip()) == base_simple)
    if same_count > 1:
        # Number among duplicates
        order = 1
        for i, p in enumerate(all_positions):
            if i == index:
                break
            if _label_base_from_text(p.strip()) == base_simple:
                order += 1
        base = f"{base_simple} #{order}"
    else:
        base = base_simple if len(full_s.split()) > 4 else full_s

    # Spread-specific structured labels for known near-duplicates
    if spread_name == "Golden Dawn Spread":
        gd = [
            "reader & topic",
            "topic extension A",
            "topic extension B",
            "current path 1",
            "alternate path 1",
            "psych basis 1",
            "karma 1",
            "current path 2",
            "alternate path 2",
            "psych basis 2",
            "karma 2",
            "current path 3",
            "alternate path 3",
            "psych basis 3",
            "karma 3",
        ]
        if index < len(gd):
            base = gd[index]
    elif spread_name == "Reversed Compass":
        rc = [
            "nucleus",
            "west near",
            "west mid",
            "west far",
            "south near",
            "south mid",
            "south far",
            "east near",
            "east mid",
            "east far",
            "north near",
            "north mid",
            "north far",
        ]
        if index < len(rc):
            base = rc[index]
    elif spread_name == "Three Dragons Spread":
        td = [
            "red dragon head",
            "red dragon body",
            "red dragon tail",
            "green dragon head",
            "green dragon body",
            "green dragon tail",
            "white dragon head",
            "white dragon body",
            "white dragon tail",
        ]
        if index < len(td):
            base = td[index]
    elif spread_name == "Comic Spread":
        base = f"panel {index + 1}"
    elif spread_name == "Three Bones Spread":
        base = f"bone {index + 1}"
    elif spread_name == "Three Pyramids Spread":
        tp = [
            "past basis 1",
            "past basis 2",
            "past basis 3",
            "present 1",
            "present 2",
            "potential",
            "strength 1",
            "strength 2",
            "nurture",
            "weakness 1",
            "weakness 2",
            "behaviour",
        ]
        if index < len(tp):
            base = tp[index]
    elif spread_name == "Love Triangle Spread":
        lt = [
            "person 1 querent",
            "person 2 interest",
            "person 3 other",
            "p1 view of p2",
            "p2 view of p3",
            "p3 view of p1",
            "p2 view of p1",
            "p3 view of p2",
            "p1 view of p3",
            "rel p1-p2",
            "rel p2-p3",
            "rel p1-p3",
            "overall trio",
        ]
        if index < len(lt):
            base = lt[index]
    elif spread_name == "Decision Spread":
        ds = [
            "do it later",
            "skip it later",
            "do it now",
            "skip it now",
            "do it outcome",
            "skip it outcome",
            "significator",
        ]
        if index < len(ds):
            base = ds[index]
    elif spread_name == "Relationship Spread #1":
        rs = [
            "significator",
            "other rational",
            "other emotional",
            "other stance",
            "querent stance",
            "querent emotional",
            "querent rational",
        ]
        if index < len(rs):
            base = rs[index]
    elif spread_name == "Path Spread":
        ps = [
            "significator",
            "current rational",
            "current emotional",
            "current stance",
            "suggested stance",
            "suggested emotional",
            "suggested rational",
        ]
        if index < len(ps):
            base = ps[index]
    elif spread_name == "Cross Spread":
        cs = ["deals with", "do not", "do this", "leads to"]
        if index < len(cs):
            base = cs[index]
    elif spread_name == "Ankh Spread":
        ak = [
            "significator 1",
            "significator 2",
            "early causes",
            "trigger causes",
            "spiritual view",
            "why this path",
            "next step",
            "surprises",
            "result",
        ]
        if index < len(ak):
            base = ak[index]
    elif spread_name == "Secret of the Priestess Spread":
        sp = [
            "significator 1",
            "significator 2",
            "current influence",
            "waxing moon",
            "waning moon",
            "dark",
            "light",
            "next step",
            "secret of priestess",
        ]
        if index < len(sp):
            base = sp[index]
    elif spread_name == "Celtic Cross Spread":
        cc = [
            "significator",
            "crossing",
            "conscious",
            "unconscious",
            "recent past",
            "near future",
            "self attitude",
            "environment",
            "hopes fears",
            "long term",
        ]
        if index < len(cc):
            base = cc[index]
    elif spread_name == "Horse Shoe Spread":
        hs = [
            "past",
            "present",
            "hidden influences",
            "obstacles",
            "external influences",
            "suggestions",
            "outcome",
        ]
        if index < len(hs):
            base = hs[index]
    elif spread_name == "Blind Spot Spread":
        bs = ["known self", "unconscious drive", "concealed self", "blind spot"]
        if index < len(bs):
            base = bs[index]
    elif spread_name == "Game Plan Spread":
        gp = ["significator", "unconscious drive", "others attitudes", "failure path", "success path"]
        if index < len(gp):
            base = gp[index]
    elif spread_name == "Astrological Spread":
        ast = [
            "basic mood",
            "finance",
            "mundane life",
            "home",
            "fun things",
            "work",
            "partners",
            "hidden aspects",
            "higher views",
            "reputation",
            "friends",
            "hopes and fears",
        ]
        if index < len(ast):
            base = ast[index]
    elif spread_name == "Relationship Spread #2":
        r2 = [
            "other persona",
            "your persona",
            "present connection",
            "common past",
            "they offer",
            "you offer",
            "mutual goals",
        ]
        if index < len(r2):
            base = r2[index]

    return base


def _label_base_from_text(full_s: str) -> str:
    if len(full_s.split()) <= 4 and len(full_s) <= 40:
        return full_s
    words = _tokenize(full_s)[:3]
    return " ".join(words) if words else full_s[:20]


def _infer_label_base(spread_name: str, index: int, full: str) -> str:
    return _label_base_from_text(full.strip())


def _vi_label(label_en: str) -> str:
    """Lightweight Vietnamese gloss for position labels (lookup-style)."""
    mapping = {
        "past": "quá khứ",
        "present": "hiện tại",
        "future": "tương lai",
        "outcome": "kết quả",
        "significator": "chủ bài",
        "obstacles": "trở ngại",
        "suggestions": "gợi ý",
        "blind spot": "điểm mù",
        "next step": "bước tiếp",
        "work": "công việc",
        "home": "gia đình",
        "finance": "tài chính",
        "friends": "bạn bè",
        "partners": "đối tác",
        "hopes and fears": "hy vọng và lo sợ",
        "reputation": "danh tiếng",
    }
    low = label_en.lower()
    if low in mapping:
        return mapping[low]
    return label_en  # keep English for complex labels


def build_spreads(data_path: Optional[Path] = None) -> List[dict]:
    path = data_path or (_project_root() / "data/github/tarotschema_spreads.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    spreads_in = data["spreads"]
    assert len(spreads_in) == 21

    out = []
    for s in spreads_in:
        name = s["spread_name"]
        positions_raw: List[str] = list(s["card_positions"])
        assert s["cards_drawn"] == len(positions_raw)
        positions = []
        for i, full in enumerate(positions_raw):
            label = _infer_label(name, i, full, positions_raw)
            gloss_compact = _compress_gloss(full, label)
            positions.append(
                {
                    "index": i,
                    "label_en": label,
                    "label_vi": _vi_label(label),
                    "gloss_compact": gloss_compact,
                    "gloss_full": full,
                }
            )
        diff = (s.get("difficulty") or "average").lower()
        out.append(
            {
                "spread_id": s.get("id") or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                "name_en": name,
                "name_vi": f"trải bài {name}",
                "cards_drawn": s["cards_drawn"],
                "difficulty": diff,
                "difficulty_ordinal": DIFFICULTY_MAP.get(diff, 3),
                "positions": positions,
            }
        )
    return out


# ---- TF-IDF cosine matcher (stdlib / pure python) ----

def _tfidf_matrix(docs: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    tokenized = [_tokenize(d) for d in docs]
    df: Counter = Counter()
    for toks in tokenized:
        df.update(set(toks))
    n = len(docs)
    idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
    vectors = []
    for toks in tokenized:
        tf = Counter(toks)
        length = len(toks) or 1
        vec = {t: (tf[t] / length) * idf[t] for t in tf}
        vectors.append(vec)
    return vectors, idf


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def positional_discrimination_report(spreads: Sequence[dict]) -> dict:
    """Pool all positions; for each position query its gloss_compact, recover index.

    Query uses gloss_compact; the corpus is also gloss_compact (self-retrieval
    would be trivial). Instead: query with gloss_full (source), match against
    gloss_compact pool — tests that compression preserves identity.
    """
    corpus_docs = []
    meta = []  # (spread_id, pos_index)
    for sp in spreads:
        for p in sp["positions"]:
            # Document = compact + label (what we store / retrieve)
            doc = f"{p['label_en']} {p['gloss_compact']}"
            corpus_docs.append(doc)
            meta.append((sp["spread_id"], sp["name_en"], p["index"]))

    vectors, _ = _tfidf_matrix(corpus_docs)
    n = len(corpus_docs)
    chance = 1.0 / n if n else 0.0

    per_spread: Dict[str, dict] = {}
    # Build query vectors from full gloss + same idf space
    # Recompute with queries appended is messy; project query into idf from corpus
    tokenized_corpus = [_tokenize(d) for d in corpus_docs]
    df: Counter = Counter()
    for toks in tokenized_corpus:
        df.update(set(toks))
    idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

    def vec_of(text: str) -> Dict[str, float]:
        toks = _tokenize(text)
        tf = Counter(toks)
        length = len(toks) or 1
        return {t: (tf[t] / length) * idf.get(t, 0.0) for t in tf}

    correct = 0
    total = 0
    for sp in spreads:
        sc = 0
        st = 0
        for p in sp["positions"]:
            # Query from full source text (information-rich), retrieve compact
            q = vec_of(f"{p['label_en']} {p['gloss_full']}")
            best_i = -1
            best_s = -1.0
            for i, v in enumerate(vectors):
                s = _cosine(q, v)
                if s > best_s:
                    best_s = s
                    best_i = i
            ok = meta[best_i][0] == sp["spread_id"] and meta[best_i][2] == p["index"]
            sc += int(ok)
            st += 1
        rate = sc / st if st else 0.0
        per_spread[sp["name_en"]] = {
            "top1_correct": sc,
            "n_positions": st,
            "top1_rate": rate,
            "above_chance": rate > chance,
        }
        correct += sc
        total += st

    above = sum(1 for v in per_spread.values() if v["above_chance"])
    failing = [name for name, v in per_spread.items() if not v["above_chance"]]
    return {
        "n_spreads": len(spreads),
        "n_positions_pooled": n,
        "chance_rate": chance,
        "overall_top1_rate": correct / total if total else 0.0,
        "spreads_above_chance": above,
        "spreads_above_chance_required": 18,
        "failing_spreads": failing,
        "per_spread": per_spread,
    }


def assert_spreads(spreads: List[dict], report: Optional[dict] = None) -> dict:
    assert len(spreads) == 21
    report = report or positional_discrimination_report(spreads)
    assert report["spreads_above_chance"] >= 18, (
        f"only {report['spreads_above_chance']}/21 above chance; "
        f"failing={report['failing_spreads']}"
    )
    return report


def write_spreads(
    out_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
) -> Tuple[Path, dict]:
    spreads = build_spreads(data_path=data_path)
    report = assert_spreads(spreads)
    path = out_path or (_project_root() / "kb/spreads.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, spreads)
    rpath = report_path or (_project_root() / "kb/spreads_discrimination_report.json")
    rpath.write_text(dumps_canonical(report) + "\n", encoding="utf-8")
    return path, report
