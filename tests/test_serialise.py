"""Serialiser determinism and compact/whitelist structural tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import (  # noqa: E402
    build_card_name_whitelist,
    build_compact_cards,
    mean_compact_tokens,
    read_jsonl,
    serialise_spine_document,
    try_load_qwen_tokenizer,
)


def test_serialiser_byte_identical_in_process():
    path = ROOT / "kb/english_spine.jsonl"
    if not path.exists():
        pytest.skip("spine not built")
    rows = read_jsonl(path)
    a = serialise_spine_document(rows)
    b = serialise_spine_document(rows)
    assert a == b
    assert a.startswith(b"[")


def test_serialiser_byte_identical_two_processes(tmp_path):
    path = ROOT / "kb/english_spine.jsonl"
    if not path.exists():
        pytest.skip("spine not built")
    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'src')!r})
from tfvn.serialise import read_jsonl, serialise_spine_document
rows = read_jsonl(Path({str(path)!r}))
sys.stdout.buffer.write(serialise_spine_document(rows))
"""
    out1 = subprocess.check_output([sys.executable, "-c", script])
    out2 = subprocess.check_output([sys.executable, "-c", script])
    assert out1 == out2
    assert len(out1) > 1000


def test_compact_156_and_token_budget():
    path = ROOT / "kb/english_spine.jsonl"
    if not path.exists():
        pytest.skip("spine not built")
    rows = read_jsonl(path)
    compact = build_compact_cards(rows)
    assert len(compact) == 156
    tok = try_load_qwen_tokenizer()
    mean = mean_compact_tokens(compact, tokenizer=tok)
    assert mean <= 65, mean


def test_whitelist_78_plus_aliases():
    wl = build_card_name_whitelist()
    assert wl["canonical_count"] == 78
    assert len(wl["canonical_names"]) == 78
    assert wl["entry_count"] >= 78
    # Magician alias present
    alias_names = {a["alias"] for a in wl["aliases"]}
    assert "the juggler" in alias_names
