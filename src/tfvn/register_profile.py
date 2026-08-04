"""Vietnamese function-word / particle frequency profile from esoteric corpus."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .serialise import dumps_canonical

# Closed-class / discourse particles diagnostic of Vietnamese register.
FUNCTION_WORDS = [
    "thì",
    "là",
    "mà",
    "của",
    "sẽ",
    "rằng",
    "và",
    "nhưng",
    "nếu",
    "khi",
    "để",
    "cho",
    "với",
    "trong",
    "ngoài",
    "đã",
    "đang",
    "bị",
    "được",
    "các",
    "những",
    "một",
    "này",
    "đó",
    "kia",
    "vì",
    "do",
    "bởi",
    "nên",
    "cũng",
    "rất",
    "nhiều",
    "ít",
    "không",
    "chưa",
    "chỉ",
    "còn",
    "hay",
    "hoặc",
    "về",
    "như",
    "theo",
    "sau",
    "trước",
    "giữa",
    "bằng",
    "từ",
    "đến",
    "tại",
    "qua",
    "lại",
    "vào",
    "ra",
    "lên",
    "xuống",
    "ạ",
    "nhé",
    "nha",
    "nhỉ",
    "ừ",
    "à",
    "chứ",
    "thôi",
    "vậy",
    "thế",
    "sao",
    "gì",
    "nào",
    "ai",
    "đâu",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _simple_vi_tokens(text: str) -> List[str]:
    """Whitespace / punctuation tokenizer (no underthesea dependency)."""
    text = unicodedata.normalize("NFC", text.lower())
    return re.findall(r"[0-9a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", text, flags=re.I)


# Core particles that MUST appear with non-zero rate in a real Vietnamese sample.
CORE_VI_FUNCTION_WORDS = ("là", "và", "của", "trong", "không", "có", "được", "cho", "với", "các")


def load_jakeveo_sample(
    data_dir: Optional[Path] = None,
    max_chars: int = 2_000_000,
    seed: int = 42,
    *,
    language: str = "vi",
) -> Tuple[str, dict]:
    """Load a sample of the Vietnamese esoteric corpus.

    Only rows with ``language == 'vi'`` (or equivalent) are ingested. Walking the
    JSONL in file order without this filter fills the budget with English TCM
    docs that appear first and produces a useless all-zero function-word profile.
    """
    root = data_dir or (
        _project_root() / "data/hf/jakeveo05__chinese-traditional-knowledge/data"
    )
    path = root / "chinese_knowledge.jsonl"
    if not path.exists():
        candidates = list(root.glob("**/*")) if root.exists() else []
        raise FileNotFoundError(f"jakeveo corpus not found at {path}; saw {candidates[:5]}")

    rng = random.Random(seed)
    vi_docs: List[dict] = []
    skipped_non_json = 0
    skipped_non_vi = 0
    total_lines = 0

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_non_json += 1
                continue
            if not isinstance(obj, dict):
                skipped_non_json += 1
                continue
            lang = (obj.get("language") or obj.get("lang") or "").strip().lower()
            cat = (obj.get("category") or "").strip().lower()
            # Accept explicit vi language, or Vietnamese-tagged categories if
            # language is missing (defensive).
            is_vi = lang == language.lower() or (
                not lang and ("vietnamese" in cat or cat.endswith("_vi") or cat == "vi")
            )
            if not is_vi:
                skipped_non_vi += 1
                continue
            text = obj.get("text") or obj.get("content") or obj.get("body") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            vi_docs.append(obj)

    if not vi_docs:
        raise RuntimeError(
            f"no Vietnamese documents found in {path} "
            f"(language={language!r}; lines={total_lines}, skipped_non_vi={skipped_non_vi})"
        )

    # Shuffle so the sample is not biased to the first few huge docs.
    rng.shuffle(vi_docs)

    chunks: List[str] = []
    total = 0
    page_markers = 0
    docs_used = 0
    for obj in vi_docs:
        text = obj.get("text") or obj.get("content") or obj.get("body") or ""
        page_markers += len(re.findall(r"---\s*Page\s+\d+\s*---", text))
        text = re.sub(r"---\s*Page\s+\d+\s*---", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        chunks.append(text)
        total += len(text)
        docs_used += 1
        if total >= max_chars:
            break

    sample = "\n".join(chunks)[:max_chars]
    # Diacritic / known-word smoke signal that the sample is Vietnamese
    has_diacritics = bool(
        re.search(
            r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
            sample,
            re.I,
        )
    )
    info = {
        "sample_chars": len(sample),
        "source_lines_total": total_lines,
        "vi_docs_available": len(vi_docs),
        "vi_docs_used": docs_used,
        "skipped_non_vi": skipped_non_vi,
        "skipped_non_json": skipped_non_json,
        "language_filter": language,
        "page_markers_stripped": page_markers,
        "has_vietnamese_diacritics": has_diacritics,
        "native_authored_fraction": "unknown",
        "note": (
            "language=='vi' only; published Vietnamese esoteric register; "
            "native vs translated fraction not partitioned."
        ),
    }
    return sample, info


def assert_corpus_profile_is_vietnamese(profile: dict, min_nonzero_fraction: float = 0.5) -> None:
    """Fail if corpus_profile looks like English contamination / empty sample."""
    corpus = profile.get("corpus_profile") or {}
    if not corpus:
        raise AssertionError("corpus_profile missing or empty")

    sample_meta = profile.get("sample") or {}
    if sample_meta.get("language_filter") not in (None, "vi"):
        # None allowed only for legacy; prefer explicit vi
        pass
    if sample_meta.get("language_filter") == "en":
        raise AssertionError("language_filter is en — must be vi")

    # Core particles must be present with positive rate
    core_hits = {w: float(corpus.get(w, 0.0)) for w in CORE_VI_FUNCTION_WORDS if w in FUNCTION_WORDS or True}
    # 'có' may not be in FUNCTION_WORDS list — check membership
    missing_core = [w for w in ("là", "và", "của", "trong", "không") if float(corpus.get(w, 0.0)) <= 0.0]
    if missing_core:
        raise AssertionError(
            f"core Vietnamese function words have zero rate (English sample?): {missing_core}"
        )

    nonzero = sum(1 for w in FUNCTION_WORDS if float(corpus.get(w, 0.0)) > 0.0)
    frac = nonzero / len(FUNCTION_WORDS)
    if frac < min_nonzero_fraction:
        raise AssertionError(
            f"only {nonzero}/{len(FUNCTION_WORDS)} function words nonzero "
            f"({frac:.0%} < {min_nonzero_fraction:.0%}) — sample not Vietnamese"
        )

    if sample_meta and sample_meta.get("has_vietnamese_diacritics") is False:
        raise AssertionError("sample has no Vietnamese diacritics")

    if sample_meta and int(sample_meta.get("sample_chars") or 0) < 10_000:
        raise AssertionError(f"sample too small: {sample_meta.get('sample_chars')} chars")


def profile_from_text(text: str) -> Dict[str, float]:
    tokens = _simple_vi_tokens(text)
    if not tokens:
        return {w: 0.0 for w in FUNCTION_WORDS}
    total = len(tokens)
    counts = Counter(tokens)
    return {w: counts.get(w, 0) / total for w in FUNCTION_WORDS}


def title_secondary_profile(vn_upright_path: Optional[Path] = None) -> Dict[str, float]:
    path = vn_upright_path or (_project_root() / "kb/vn_upright.jsonl")
    if not path.exists():
        return {}
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("vi_provenance") == "source" and row.get("title_secondary"):
            texts.append(row["title_secondary"])
    return profile_from_text("\n".join(texts))


def build_register_profile(
    max_chars: int = 2_000_000,
    vn_upright_path: Optional[Path] = None,
) -> dict:
    sample, info = load_jakeveo_sample(max_chars=max_chars, language="vi")
    corpus_profile = profile_from_text(sample)
    sec_profile = title_secondary_profile(vn_upright_path)
    profile = {
        "segmenter": "simple_regex_vi_tokens",
        "segmenter_version": "1.0",
        "function_words": FUNCTION_WORDS,
        "corpus_profile": corpus_profile,
        "title_secondary_profile": sec_profile,
        "sample": info,
        "token_count_sample": len(_simple_vi_tokens(sample)),
        "core_vi_rates": {
            w: float(corpus_profile.get(w, 0.0))
            for w in ("là", "và", "của", "trong", "không", "được", "cho", "với", "các", "của")
        },
    }
    assert_corpus_profile_is_vietnamese(profile)
    return profile


def write_register_profile(out_path: Optional[Path] = None, max_chars: int = 2_000_000) -> Path:
    profile = build_register_profile(max_chars=max_chars)
    path = out_path or (_project_root() / "kb/vn_register_profile.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_canonical(profile) + "\n", encoding="utf-8")
    return path
