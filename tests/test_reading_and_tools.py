import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tfvn.reading import (
    N_CTX,
    assemble_prefix,
    assemble_prompt,
    card_block,
    truncate_history,
)
from tfvn.tools import Deck, ToolBox, ClarificationManager

DRAW = [
    {"card_id": 0, "name_en": "The Fool", "orientation": "upright"},
    {"card_id": 5, "name_en": "The Hierophant", "orientation": "reversed"},
    {"card_id": 42, "name_en": "Six of Swords", "orientation": "upright"}]


def _compact():
    from tfvn.reading import load_compact_index

    return load_compact_index()


def test_deck_deterministic_seed_reproducible():
    d = Deck()
    a = d.draw(10, seed=123)
    b = d.draw(10, seed=123)
    assert a == b
    c = d.draw(10, seed=124)
    assert a != c
    assert all(0 <= x["card_id"] < 78 for x in a)
    assert {x["orientation"] for x in a} <= {"upright", "reversed"}
    full = d.shuffle(0)
    assert sorted(full) == list(range(78))


def test_assemble_prompt_byte_stable():
    p1 = assemble_prompt("Công việc của tôi?", DRAW, ["quá khứ", "hiện tại", "tương lai"])
    p2 = assemble_prompt("Công việc của tôi?", DRAW, ["quá khứ", "hiện tại", "tương lai"])
    assert p1 == p2
    # dict key order must not leak into output
    shuffled = [{k: v for k, v in reversed(list(d.items()))} for d in DRAW]
    p3 = assemble_prompt("Công việc của tôi?", shuffled, ["quá khứ", "hiện tại", "tương lai"])
    assert p3 == p1


def test_prefix_independent_of_question():
    prefix = assemble_prefix(DRAW, ["a", "b", "c"])
    q1 = assemble_prompt("hỏi một", DRAW, ["a", "b", "c"])
    q2 = assemble_prompt("hỏi hai", DRAW, ["a", "b", "c"])
    assert q1.startswith(prefix + "\n") or q1.split("\n---\n")[:-1] == \
        q2.split("\n---\n")[:-1]
    assert prefix in q1 and prefix in q2


def test_card_block_unknown_card_raises():
    import pytest

    bad = [{"card_id": 999, "name_en": "?", "orientation": "upright"}]
    with pytest.raises(KeyError):
        card_block(bad)


def test_truncate_history_drops_oldest_first():
    hist = [{"content": f"x{i} " * 50} for i in range(6)]
    kept = truncate_history(hist, token_budget=200, tokenizer_len=lambda s: len(s))
    assert kept and kept[-1] is not hist[0]
    assert [m["content"] for m in kept][-1] == hist[-1]["content"]
    assert len(kept) < len(hist)


def test_n_ctx_documented():
    assert N_CTX == 4096


# ------------------------------------------------------------------ tools ---

def test_toolbox_draw_and_meaning():
    tb = ToolBox()
    r = tb.execute("draw_cards", {"n": 5, "seed": 99})
    assert len(r["cards"]) == 5
    m = tb.execute("get_card_meaning",
                   {"card_id": r["cards"][0]["card_id"],
                    "orientation": r["cards"][0]["orientation"]})
    assert m["keywords_en"]
    unknown = tb.execute("get_card_meaning", {"card_id": 0, "orientation": "sideways"})
    assert "error" in unknown
    bogus = tb.execute("no_such_tool", {})
    assert "error" in bogus


def test_clarification_gate():
    assert ClarificationManager.needs_clarification("", None)
    assert ClarificationManager.needs_clarification("Bói đi", None)
    assert not ClarificationManager.needs_clarification(
        "Tình yêu của tôi và người ấy sẽ ra sao 3 tháng tới?", 3)
    out = ClarificationManager.clarify()
    assert "lĩnh vực" in out
