import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evals.vn_tones import (
    apply_tone,
    build_items,
    build_vocab,
    decompose,
    format_prompt,
    score_items,
    tone_variants,
)
from evals.provider import MockProvider


def test_decompose_plan_examples():
    assert decompose("má")[0] == "ma" and decompose("má")[1] == "sac"
    assert decompose("mà")[0] == "ma" and decompose("mà")[1] == "huyen"
    assert decompose("bàn")[1] == "huyen" and decompose("bán")[1] == "sac"
    assert decompose("cây")[1] is None  # ngang


def test_apply_tone_roundtrip():
    for s in ("má", "bàn", "khoẻ", "người", "sức"):
        base, _, _ = decompose(s)
        outs = {apply_tone(base, t) for t in ("sac", "huyen", "hoi", "nga", "nang")}
        assert len(outs) == 5, f"{s}: {outs}"


def test_variants_of_ma_are_real_syllables():
    assert set(tone_variants("má")) == {"mà", "mả", "mã", "mạ"}


def test_build_vocab_filters_nonalpha():
    v = build_vocab(["xin chào, bạn 123 abc-là"])
    assert "xin" in v and "chào" in v and "bạn" in v
    assert "123" not in v
    assert all(re.fullmatch(r"[a-zà-ỹđ]+", w) for w in v)


def test_build_items_shape_and_labels(tmp_path):
    items = build_items(ROOT / "kb/cards.jsonl", n_items=25, seed=7)
    assert len(items) <= 25
    for it in items:
        assert len(it["options"]) == 4
        assert len(set(it["options"])) == 4  # no duplicate options
        assert it["options"][it["answer_idx"]] == it["answer_word"]
        assert "_____" in it["blanked"]
    again = build_items(ROOT / "kb/cards.jsonl", n_items=25, seed=7)
    assert items == again  # deterministic


def test_format_prompt_contains_options():
    items = build_items(ROOT / "kb/cards.jsonl", n_items=5, seed=1)
    p = format_prompt(items[0])
    for i, o in enumerate(items[0]["options"]):
        assert f"{'ABCD'[i]}. {o}" in p


def test_score_letter_parse_mock():
    class Fixed(MockProvider):
        def __init__(self, letter):
            super().__init__([])
            self.letter = letter
            self.calls = 0

        def generate(self, prompt, **kw):
            self.calls += 1
            return self.letter

    items = build_items(ROOT / "kb/cards.jsonl", n_items=10, seed=3)
    rep = score_items(Fixed("A"), items)
    assert rep["scoring"] == "letter-parse"
    assert rep["n_items"] == 10 and 0.0 <= rep["accuracy"] <= 1.0
