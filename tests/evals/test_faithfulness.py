import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evals.faithfulness import (
    enforce_floor,
    format_prompt,
    gate,
    make_cases,
    run_gate,
)


class Scripted:
    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def generate(self, prompt, **kw):
        return self.outputs.pop(0)


def _kb():
    from tfvn.serialise import read_jsonl

    return read_jsonl(ROOT / "kb/cards.jsonl")


def test_make_cases_stratified_and_seeded():
    kb = _kb()
    cases = make_cases(kb, n_per_stratum=4, seed=1)
    by = {}
    for c in cases:
        by.setdefault(c["stratum"], []).append(c)
    assert set(by) == {"single", "three", "ten"}
    assert all(len(v) == 4 for v in by.values())
    assert all(len(c["draw"]) == c["size"] for c in cases)
    assert all(c["positions"] and len(c["positions"]) == c["size"] for c in cases)
    again = make_cases(kb, n_per_stratum=4, seed=1)
    assert cases == again


def test_format_prompt_mentions_draw_and_positions():
    kb = _kb()
    case = make_cases(kb, n_per_stratum=1, seed=0)[0]
    p = format_prompt(case)
    for d in case["draw"]:
        assert d["name_en"] in p
    for pos in case["positions"]:
        assert pos in p


def test_gate_passes_faithful_output():
    kb = _kb()
    from tfvn.validators import load_whitelist

    wl = load_whitelist()
    case = make_cases(kb, n_per_stratum=1, seed=5)[0]
    parts = [f"Vị trí {pos}: lá {d['name_en']} "
             f"{'xuôi' if d['orientation'] == 'upright' else 'đảo'} "
             "cho thấy một thông điệp quan trọng."
             for pos, d in zip(case["positions"], case["draw"])]
    out = " ".join(parts)
    r = gate(out, case, kb, wl)
    assert r["passed"], r["failures"]


def test_gate_catches_crafted_hallucination():
    """Plan QA: inject 'The Fool' into a reading that drew other cards only."""
    kb = _kb()
    from tfvn.validators import load_whitelist

    wl = load_whitelist()
    cases = make_cases(kb, n_per_stratum=10, seed=7)
    case = next(c for c in cases if c["size"] >= 3
                and all(d["card_id"] != 0 for d in c["draw"]))
    good_part = " ".join(
        f"Vị trí {pos}: lá {d['name_en']} mang ý nghĩa riêng."
        for pos, d in zip(case["positions"], case["draw"]))
    baited = good_part + " Ngoài ra The Fool cũng xuất hiện trong quá khứ."
    r = gate(baited, case, kb, wl)
    assert not r["passed"]
    assert any("hallucinated" in f for f in r["failures"])


def test_gate_catches_missing_position_and_orientation():
    kb = _kb()
    from tfvn.validators import load_whitelist

    wl = load_whitelist()
    rows = {(r["card_id"], r["orientation"]): r for r in kb}
    cases = make_cases(kb, n_per_stratum=20, seed=11)
    case = next(c for c in cases if c["size"] == 3
                and all((d["card_id"], "upright") in rows and
                        (rows[(d["card_id"], "upright")].get("keywords_en"))
                        for d in c["draw"] if d["orientation"] == "reversed")
                and any(d["orientation"] == "reversed" for d in c["draw"]))
    # omit the last position entirely + describe reversed card with upright keywords
    up_kws = [k for k in rows[(case['draw'][2]['card_id'], 'upright')]["keywords_en"]]
    rev_name = case["draw"][2]["name_en"]
    text = (f"Vị trí {case['positions'][0]}: lá {case['draw'][0]['name_en']} rõ ràng. "
            f"Vị trí {case['positions'][1]}: lá {case['draw'][1]['name_en']} cũng vậy. "
            f"Lá {rev_name}: {', '.join(up_kws[:3])}.")
    r = gate(text, case, kb, wl)
    assert not r["passed"]
    kinds = " | ".join(r["failures"])
    assert "missing_positions" in kinds or "orientation" in kinds


def test_run_gate_end_to_end_scripted(tmp_path):
    kb_path = ROOT / "kb/cards.jsonl"

    class Faithful:
        name = "faithful-mock"

        def generate(self, prompt, **kw):
            # parse card names from the prompt itself and echo them per position
            import re

            names = re.findall(r"([A-Z][a-zA-Z]+(?: of [A-Z][a-zA-Z]+| [A-Z][a-zA-Z]+)*?) \(", prompt)
            pos = re.findall(r"Các vị trí: (.+?)\. Câu hỏi", prompt)[0].split(", ")
            return " ".join(f"Vị trí {p.strip()}: lá {n} có ý nghĩa." for p, n in zip(pos, names))

    rep = run_gate(Faithful(), kb_path, n_per_stratum=2, seed=3)
    assert rep["n_cases"] == 6
    assert set(rep["by_stratum"]) == {"single", "three", "ten"}


def test_enforce_floor_rules():
    base = {"by_stratum": {"single": {"pass_rate": 0.9},
                           "ten": {"pass_rate": 0.4}}}
    ft = {"by_stratum": {"single": {"pass_rate": 0.95},
                         "ten": {"pass_rate": 0.55}}}
    chk = enforce_floor(ft, base)
    worse = {"by_stratum": {"single": {"pass_rate": 0.5},
                            "ten": {"pass_rate": 0.55}}}
    chk2 = enforce_floor(worse, base)
    assert not chk2["meets_all_floors"]     # single regressed below base 0.9
