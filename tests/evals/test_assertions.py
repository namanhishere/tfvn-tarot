import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest
from evals.assertions import run_case, run_suite
from evals.provider import MockProvider


class ScriptedProvider:
    """Returns queued outputs in order; records prompts."""

    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


@pytest.fixture(scope="module")
def kb():
    from tfvn.serialise import read_jsonl

    return read_jsonl(ROOT / "kb/cards.jsonl")


CASE_BASE = {
    "case_id": "t",
    "category": "reading",
    "prompt_vi": "đọc bài",
    "draw": [{"card_id": 0, "name_en": "The Fool", "orientation": "upright"}],
    "positions": [],
}


def test_contains_all_cards_pass_fail(kb):
    case = {**CASE_BASE,
            "assertions": [{"type": "contains_all_cards"}]}
    ok = run_case(ScriptedProvider(["Lá The Fool mang ý nghĩa khởi đầu mới."]),
                  case, kb_rows=kb, whitelist=None)
    assert ok["passed"] and not ok["failures"]
    bad = run_case(ScriptedProvider(["Một lá bài khác."]), case, kb_rows=kb, whitelist=None)
    assert not bad["passed"] and "missing cards" in bad["failures"][0]


def test_hallucination_bait_detection(kb):
    case = {**CASE_BASE,
            "assertions": [{"type": "contains_none", "forbidden": ["The Magician"]}]}
    bad = run_case(ScriptedProvider(["The Fool và The Magician đều xuất hiện."]),
                   case, kb_rows=kb, whitelist=None)
    assert not bad["passed"]


def test_containment_catches_faux_major(kb):
    case = {**CASE_BASE, "assertions": [{"type": "containment_ok"}]}
    bad = run_case(ScriptedProvider(["The Fool bên cạnh The Nonexistent Card."]),
                   case, kb_rows=kb, whitelist=None)
    assert not bad["passed"], "faux major must be flagged"


def test_orientation_ok_reversed_upright_keywords(kb):
    rows = {(r["card_id"], r["orientation"]): r for r in kb}
    # pick a reversed row with upright keywords available
    rev = next(r for r in kb if r["orientation"] == "reversed"
               and (r["card_id"], "upright") in rows and r.get("keywords_en"))
    up_kws = [k for k in rows[(rev["card_id"], "upright")].get("keywords_en") or []]
    text_with_kws = f"Lá {rev['name_en']} đảo: {' '.join(up_kws[:3])}."
    case = {"case_id": "o", "category": "reading", "prompt_vi": "?",
            "draw": [{"card_id": rev["card_id"], "name_en": rev["name_en"],
                      "orientation": "reversed"}],
            "positions": [],
            "assertions": [{"type": "orientation_ok"}]}
    bad = run_case(ScriptedProvider([text_with_kws]), case, kb_rows=kb, whitelist=None)
    assert not bad["passed"]
    good = run_case(ScriptedProvider(["Ý nghĩa của lá này bị chặn lại, trì hoãn."]),
                    case, kb_rows=kb, whitelist=None)
    assert good["passed"]


def test_mentions_positions(kb):
    case = {**CASE_BASE, "positions": ["quá khứ", "hiện tại"],
            "assertions": [{"type": "mentions_positions"}]}
    good = run_case(ScriptedProvider(["Vị trí quá khứ... còn hiện tại..."]), case,
                    kb_rows=kb, whitelist=None)
    bad = run_case(ScriptedProvider(["Chỉ nói về quá khứ thôi."]), case,
                   kb_rows=kb, whitelist=None)
    assert good["passed"] and not bad["passed"]


def test_refusal_and_regex_and_len(kb):
    case = {**CASE_BASE, "category": "safety",
            "assertions": [{"type": "refusal_form"}, {"type": "min_words", "n": 5}]}
    good = run_case(ScriptedProvider(
        ["Mình không thể thay bác sĩ. Bạn nên đi khám; thẻ bài chỉ mang tính gợi mở."]),
        case, kb_rows=kb, whitelist=None)
    bad = run_case(ScriptedProvider(["Uống thuốc đi."]), case, kb_rows=kb, whitelist=None)
    assert good["passed"] and not bad["passed"]

    case2 = {**CASE_BASE, "assertions": [
        {"type": "regex", "pattern": r"\d{3}", "negate": True}]}
    r = run_case(ScriptedProvider(["Gọi 911 ngay."]), case2, kb_rows=kb, whitelist=None)
    assert not r["passed"]


def test_unknown_assertion_type_raises(kb):
    case = {**CASE_BASE, "assertions": [{"type": "nope"}]}
    with pytest.raises(ValueError):
        run_case(ScriptedProvider(["x"]), case, kb_rows=kb, whitelist=None)


def test_run_suite_end_to_end(tmp_path, kb):
    suite = tmp_path / "suite.jsonl"
    cases = [
        {**CASE_BASE, "case_id": "c1",
         "assertions": [{"type": "contains_all_cards"}]},
        {**CASE_BASE, "case_id": "c2", "category": "safety",
         "assertions": [{"type": "refusal_form"}]},
    ]
    suite.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases),
                     encoding="utf-8")
    p = ScriptedProvider(["Lá The Fool xuất hiện ở đây.",
                          "Mình không thể thay thế chuyên gia y tế."])
    rep = run_suite(p, suite, kb_path=ROOT / "kb/cards.jsonl")
    assert rep["n_cases"] == 2 and rep["n_passed"] == 2
    assert rep["by_category"]["safety"]["passed"] == 1
