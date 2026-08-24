"""W7.5: extensibility conformance — synthetic 3-card fake deck.

Proves the full KB path works for ANY registered deck without code changes:
  registration (kb/decks/*.json) -> whitelist extension -> compact-card
  rendering -> byte-stable prompt assembly -> validators accept the fake names.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


import pytest

from tfvn.reading import assemble_prompt, card_block
from tfvn.validators import (
    load_whitelist,
    validate_card_name_containment,
    validate_orientation_consistency,
)

FIXTURE = ROOT / "tests/fixtures/fake_deck_compact.jsonl"
FAKE_NAMES = ["The Algorithm", "The Garden", "The Mirror"]
DRAW = [
    {"card_id": 100, "name_en": "The Algorithm", "orientation": "upright"},
    {"card_id": 101, "name_en": "The Garden", "orientation": "reversed"},
    {"card_id": 102, "name_en": "The Mirror", "orientation": "upright"}]


@pytest.fixture()
def registered_deck():
    """Temporarily register the fake deck through kb/decks/ (the same path a
    real deck would use), then clean up."""
    decks_dir = ROOT / "kb" / "decks"
    decks_dir.mkdir(parents=True, exist_ok=True)
    reg = decks_dir / "fake_deck.json"
    reg.write_text(json.dumps({
        "deck": "fake_deck",
        "canonical_names": FAKE_NAMES,
        "compact_cards": str(FIXTURE),
    }), encoding="utf-8")
    yield reg
    shutil.rmtree(decks_dir, ignore_errors=True)


def test_fixture_rows_valid_compact_schema():
    rows = [json.loads(l) for l in FIXTURE.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    for r in rows:
        assert r["orientation"] in ("upright", "reversed")
        assert r["keywords_en"] and r["name_en"] in FAKE_NAMES


def test_registered_extension_extends_whitelist(registered_deck):
    wl = load_whitelist()
    for n in FAKE_NAMES:
        assert n in wl["canonical_names"], f"{n} not merged from {registered_deck.name}"
    # containment validator now accepts fake names in prose
    r = validate_card_name_containment(
        "Lá The Algorithm và The Garden cùng xuất hiện.", wl)
    assert not r["hallucinated_like"], "fake deck names must not be flagged as faux majors"


def test_fake_deck_full_path_conformance(registered_deck):
    """draw -> byte-stable prompt -> validators accept."""
    wl = load_whitelist()

    from tfvn.serialise import read_jsonl

    idx = {(int(r["card_id"]), r["orientation"]): r
           for r in read_jsonl(FIXTURE)}

    # byte-stable prompt over fake cards
    p1 = assemble_prompt("Hãy đọc bài cho dự án mới của tôi?", DRAW,
                         ["hiện tại", "thách thức", "kết quả"], compact=idx)
    p2 = assemble_prompt("Hãy đọc bài cho dự án mới của tôi?", DRAW,
                         ["hiện tại", "thách thức", "kết quả"], compact=idx)
    assert p1 == p2
    for d in DRAW:
        assert d["name_en"] in p1

    # validators accept a faithful fake-deck reading
    output = ("Vị trí hiện tại: lá The Algorithm cho thấy một quy luật rõ ràng. "
              "Vị trí thách thức: lá The Garden đảo ngược cảnh báo sự bùng phát. "
              "Vị trí kết quả: lá The Mirror mang lại sự rõ ràng.")
    c = validate_card_name_containment(
        output, wl, required_names=[d["name_en"] for d in DRAW])
    assert not c["not_in_whitelist"] and not c["hallucinated_like"], c
    o = validate_orientation_consistency(output, DRAW, [])
    assert o["ok"], o["violations"]


def test_card_block_renders_fake_cards():
    from tfvn.serialise import read_jsonl

    idx = {(int(r["card_id"]), r["orientation"]): r
           for r in read_jsonl(FIXTURE)}
    block = card_block(DRAW, idx)
    for d in DRAW:
        assert d["name_en"] in block
