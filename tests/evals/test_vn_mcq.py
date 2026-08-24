import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from evals.vn_mcq import build_mcq, format_prompt, score_mcq
from evals.provider import MockProvider


@pytest.fixture(scope="module")
def items():
    return build_mcq(ROOT / "kb/cards.jsonl", n_items=40, seed=42)


def test_build_deterministic(items):
    again = build_mcq(ROOT / "kb/cards.jsonl", n_items=40, seed=42)
    assert items == again


def test_item_shape(items):
    assert len(items) == 40
    for it in items:
        assert len(it["options"]) == 4
        assert len(set(it["options"])) == 4
        assert it["options"][it["answer_idx"]] == it["answer_name"]
        assert len(it["context_vi"].split()) >= 5
    ids = {i["item_id"] for i in items}
    assert len(ids) == len(items)


def test_prompt_format(items):
    p = format_prompt(items[0])
    assert "A." in p and "D." in p
    assert items[0]["context_vi"] in p


def test_score_letter_parse_mock(items):
    # mock replies cycle; craft replies that always answer with the right letter is
    # impossible without seeing the prompt -> instead verify determinism + shape.
    class FixedLetter(MockProvider):
        def __init__(self, letter):
            super().__init__([])
            self.letter = letter
            self.calls = 0

        def generate(self, prompt, **kw):
            self.calls += 1
            return self.letter

    p = FixedLetter("A")
    rep = score_mcq(p, items[:10])
    assert rep["scoring"] == "letter-parse"
    assert rep["n_items"] == 10
    assert p.calls == 10
    acc_again = score_mcq(FixedLetter("A"), items[:10])
    assert acc_again["accuracy"] == rep["accuracy"]
