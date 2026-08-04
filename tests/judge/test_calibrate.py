"""W0.4 tests: taxonomy schema validity + degradation/harness behaviour."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from judge.calibrate import (  # noqa: E402
    MockJudge,
    _build_corpus,
    degrade_faithfulness,
    degrade_orientation,
    degrade_tone,
    degrade_translationese,
    run,
    wilson_ci,
)

TAXONOMY = json.loads((ROOT / "judge" / "taxonomy.json").read_text())


def test_taxonomy_has_four_axes():
    assert set(TAXONOMY["axes"]) == {"tone", "orientation", "translationese", "faithfulness"}


def test_each_axis_has_chance_baseline_and_floor():
    for name, ax in TAXONOMY["axes"].items():
        assert 0 < ax["chance_baseline"] < 1, name
        assert ax["acceptance_floor"].startswith("ci_lower_bound >"), name
        assert "detection_criterion" in ax and "degradation" in ax, name


def test_secondary_judge_is_unrelated_architecture():
    sec = TAXONOMY["judges"]["secondary"]
    assert sec["provider"] != TAXONOMY["judges"]["primary"]["provider"]
    assert "unrelated_architecture_reason" in sec
    assert sec["provisioned"] is False


def test_no_gate_starts_empty():
    assert TAXONOMY["no_gate"] == []


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(50, 100)
    assert abs(lo - 0.402) < 0.002
    assert abs(hi - 0.598) < 0.002


def test_wilson_ci_zero_n():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_degrade_tone_substitutes_valid_pair():
    text = "má tôi bàn việc này"
    rng = random.Random(7)
    out, flagged = degrade_tone(text, rng)
    assert flagged is True
    assert out != text
    assert "bán" in out or "mà" in out or "mạ" in out


def test_degrade_tone_no_hits_is_clean():
    out, flagged = degrade_tone("không có từ hợp lệ ở đây", random.Random(7))
    assert flagged is False
    assert out == "không có từ hợp lệ ở đây"


def test_degrade_orientation_only_on_reversed():
    out, flagged = degrade_orientation("the card means joy upright", random.Random(7))
    assert flagged is False


def test_degrade_translationese_calques_clause():
    out, flagged = degrade_translationese("Tôi tin rằng anh ấy đến", random.Random(7))
    assert flagged is True
    assert "rằng" in out
    assert out.startswith("anh ấy đến")


def test_harness_runs_and_marks_axes():
    rng = random.Random(42)
    rates = {"tone": 0.9, "orientation": 0.9, "translationese": 0.9, "faithfulness": 0.9}
    judges = [MockJudge(rates, rng), MockJudge(rates, rng)]
    data = run(judges, TAXONOMY, rng)
    results = data["results"]
    assert set(results) == set(TAXONOMY["axes"])
    for axis, res in results.items():
        assert len(res["judges"]) == 2
        for j in res["judges"]:
            assert j["n"] > 0
            assert 0 <= j["rate"] <= 1
            assert j["ci_low"] <= j["rate"] <= j["ci_high"]


def test_high_rate_axis_passes_low_rate_axis_no_gates():
    rng = random.Random(3)
    rates = {"tone": 0.95, "orientation": 0.95, "translationese": 0.95, "faithfulness": 0.95}
    judges = [MockJudge(rates, rng)]
    data = run(judges, TAXONOMY, rng)
    assert all(r["passes"] for r in data["results"].values())
    assert data["taxonomy"]["no_gate"] == []


def test_build_corpus_produces_n_samples():
    rng = random.Random(42)
    corpus = _build_corpus("tone", 20, rng)
    assert len(corpus) == 20
    assert all(isinstance(t, str) and isinstance(b, bool) for t, b in corpus)


def test_degrade_faithfulness_swaps_meaning():
    text = "The Fool means new beginnings"
    out, flagged = degrade_faithfulness(text, random.Random(11))
    assert isinstance(out, str)
    assert isinstance(flagged, bool)
