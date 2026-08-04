"""Register profile must be built from Vietnamese docs, not English TCM prefix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.register_profile import (  # noqa: E402
    FUNCTION_WORDS,
    assert_corpus_profile_is_vietnamese,
    build_register_profile,
    load_jakeveo_sample,
    profile_from_text,
)


def test_load_jakeveo_sample_filters_vi_only():
    sample, info = load_jakeveo_sample(max_chars=200_000, language="vi")
    assert info["language_filter"] == "vi"
    assert info["vi_docs_used"] >= 1
    assert info["skipped_non_vi"] >= 1  # corpus has English docs that must be skipped
    assert info["sample_chars"] >= 10_000
    assert info["has_vietnamese_diacritics"] is True
    # Sample itself should contain common Vietnamese function words
    low = sample.lower()
    for w in ("là", "và", "của", "trong"):
        assert w in low, f"expected {w!r} in Vietnamese sample text"


def test_build_register_profile_nonzero_core_particles():
    # Drive the real builder (may take a few seconds on the full corpus filter)
    profile = build_register_profile(max_chars=500_000)
    assert_corpus_profile_is_vietnamese(profile)
    corpus = profile["corpus_profile"]
    for w in ("là", "và", "của", "trong", "không"):
        assert corpus[w] > 0.0, f"{w} rate is zero: {corpus[w]}"
    nonzero = sum(1 for w in FUNCTION_WORDS if corpus.get(w, 0.0) > 0.0)
    assert nonzero >= len(FUNCTION_WORDS) // 2
    assert profile["sample"]["language_filter"] == "vi"


def test_assert_rejects_english_contaminated_profile():
    english = (
        "The patient presented with chronic fatigue and liver qi stagnation. "
        "Treatment included acupuncture and herbal formulas according to TCM theory. "
        "The practitioner noted significant improvement after several sessions."
    )
    bad = {
        "corpus_profile": profile_from_text(english),
        "sample": {
            "language_filter": "vi",  # lying filter
            "sample_chars": len(english),
            "has_vietnamese_diacritics": False,
        },
    }
    with pytest.raises(AssertionError):
        assert_corpus_profile_is_vietnamese(bad)


def test_shipped_kb_profile_is_vietnamese():
    path = ROOT / "kb/vn_register_profile.json"
    if not path.exists():
        pytest.skip("profile not built")
    prof = json.loads(path.read_text(encoding="utf-8"))
    assert_corpus_profile_is_vietnamese(prof)
    assert prof.get("sample", {}).get("language_filter") == "vi"
    for w in ("là", "và", "của", "trong", "không"):
        assert prof["corpus_profile"][w] > 0.0
