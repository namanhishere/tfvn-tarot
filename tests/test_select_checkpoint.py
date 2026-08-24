import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from select_checkpoint import eligibility, load_runs, select


RUN_OK = {"run": "sft_r32_core", "held_out_loss": 1.9, "judge_win_rate": 0.62,
          "orientation_jaccard_rate": 0.03, "tone_accuracy": 0.78,
          "harmful_refusal": 0.96, "benign_refusal": 0.08}


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_eligibility_hard_gates():
    ok, why = eligibility(RUN_OK, tone_floor=0.5)
    assert ok and not why
    bad = {**RUN_OK, "orientation_jaccard_rate": 0.30, "tone_accuracy": 0.4}
    ok2, why2 = eligibility(bad, tone_floor=0.5)
    assert not ok2 and len(why2) == 2


def test_select_prefers_best_composite(tmp_path):
    better = {**RUN_OK, "run": "r32", "held_out_loss": 1.7, "judge_win_rate": 0.70}
    worse = {**RUN_OK, "run": "r16", "held_out_loss": 2.4, "judge_win_rate": 0.55}
    res = select([worse, better])
    assert res["selected"] == "r32"
    assert res["ranked"][0]["run"] == "r32"


def test_ineligible_runs_excluded_even_if_composite_better(tmp_path):
    good = {**RUN_OK, "run": "good"}
    cheat = {**RUN_OK, "run": "cheater", "judge_win_rate": 0.95,
             "orientation_jaccard_rate": 0.5}   # fails hard gate
    res = select([cheat, good])
    assert res["selected"] == "good"
    assert any(not r["eligible"] and r["run"] == "cheater" for r in res["ranked"])


def test_core_wins_tie_against_bulk(tmp_path):
    core = {**RUN_OK, "run": "a_core"}
    bulk = {**RUN_OK, "run": "b_bulk"}
    # identical composites: same numbers, different names/tiers
    res = select([bulk, core])
    assert res["selected"] == "a_core"
    assert "core-only" in res["reason"]


def test_all_fail_returns_none():
    dead = {**RUN_OK, "run": "dead", "tone_accuracy": 0.1}
    res = select([dead])
    assert res["selected"] is None
    assert "no eligible" in res["reason"]


def test_cli_end_to_end(tmp_path):
    p = _write(tmp_path, "run.json", RUN_OK)
    out = tmp_path / "sel"
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/select_checkpoint.py"),
         "--runs", str(p), "--out", str(out)],
        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    md = out.with_suffix(".md").read_text()
    assert "Selected:" in md and "sft_r32_core" in md
    js = json.loads(out.with_suffix(".json").read_text())
    assert js["selected"] == "sft_r32_core"
    assert js["pre_safety_baseline"]["harmful_refusal"] == 0.96
