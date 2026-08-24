import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.drift import (
    analyse_generation,
    classify_tokens,
    compare,
    load_whitelist_surfaces,
    mask_card_names,
    run as run_drift,
    whitelist_hash,
)


class FixedGen:
    name = "fixed"

    def __init__(self, texts):
        self.texts = list(texts)

    def generate(self, prompt, **kw):
        return self.texts.pop(0)


WL = ROOT / "kb/card_name_whitelist.json"


def test_whitelist_loads_and_hashes():
    surfaces = load_whitelist_surfaces(WL)
    assert "The Fool" in surfaces and len(surfaces) >= 78
    assert len(whitelist_hash(WL)) == 64


def test_mask_card_names_removes_terminology():
    masked = mask_card_names("Lá The Fool và Queen of Cups xuất hiện.",
                             ["The Fool", "Queen of Cups"])
    assert "The Fool" not in masked and "Queen" not in masked


def test_classify_tokens():
    cls = classify_tokens("tình yêu của The Fool là beautiful 123!")
    # "Anh" is pure ASCII -> conservatively 'en'; that ambiguity is exactly why
    # the metric uses delta-from-baseline instead of absolute ratios.
    assert classify_tokens("không có tiếng Anh ở đây") == \
        ["vi", "vi", "vi", "en", "vi", "vi"]


def test_analyse_detects_collapse_after_300():
    vi = "bài tarot nói về tình yêu và công việc của bạn sẽ thay đổi "
    en = "the cards tell about your love and work in the future now "
    surf = load_whitelist_surfaces(WL)
    good = (vi * 20) + "The Fool"
    bad = vi * 12 + en * 22 + vi * 10   # EN tail lands after token ~300
    r_good = analyse_generation(good, surf, collapse_threshold=0.6)
    r_bad = analyse_generation(bad, surf, collapse_threshold=0.6)
    assert r_good["collapses"] == 0
    assert r_bad["collapses"] >= 1
    assert r_bad["mean_vi_frac"] < r_good["mean_vi_frac"]


def test_compare_flags_regression_and_hash_mismatch():
    base = {"aggregate": {"mean_vi_frac": 0.95, "collapse_rate": 0.0},
            "whitelist_sha256": "aaa"}
    cand = {"aggregate": {"mean_vi_frac": 0.60, "collapse_rate": 0.4},
            "whitelist_sha256": "bbb"}
    c = compare(base, cand)
    assert c["regressed"] and not c["whitelist_match"]
    same = compare(base, {**cand, "whitelist_sha256": "aaa"})
    assert same["whitelist_match"]


def test_run_end_to_end_fixed_provider():
    surf = load_whitelist_surfaces(WL)
    vi_text = "lá bài nói về sự khởi đầu mới trong cuộc sống của bạn "
    p = FixedGen([vi_text * 20] * 4)
    rep = run_drift(p, n_gens=4, max_tokens=700, surfaces=surf,
                    collapse_threshold=0.5, seed=1)
    assert rep["n_generations"] == 4
    assert rep["aggregate"]["mean_vi_frac"] > 0.9
    assert rep["aggregate"]["collapse_rate"] == 0.0
