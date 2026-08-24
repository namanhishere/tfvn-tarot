import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_sft import (
    OrientationTripwire,
    build_tokenized,
    format_example,
    load_rows,
    pick_dtype,
)

ROW = {
    "task_type": "reading",
    "spread_name_vi": "trải bài 3 lá",
    "question_vi": "Công việc của tôi thế nào?",
    "cards_used": [
        {"card_id": 0, "name_en": "The Fool", "orientation": "upright",
         "polarity_axis": None},
        {"card_id": 5, "name_en": "The Hierophant", "orientation": "reversed",
         "polarity_axis": "delayed"},
    ],
    "position_glosses": ["quá khứ", "hiện tại"],
    "reading_vi": "Lá The Fool cho thấy một khởi đầu mới...",
}
SAFETY_ROW = {**ROW, "task_type": "safety",
              "reading_vi": "Mình không thể thay bác sĩ..."}
CORRECTION_ROW = {**ROW, "task_type": "correction",
                  "wrong_claim": "The Fool là lá Two of Cups"}


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1

    class _Enc:
        def __init__(self, ids):
            self.input_ids = ids

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "PROMPT|" if add_generation_prompt else "PROMPT|" + msgs[-1]["content"] + "|END"

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        # deterministic fake tokenization: 1 char = 1 token, '|' = 2
        ids = [10 + (ord(c) % 20) for c in text]
        if return_tensors == "pt":
            return {"input_ids": ids}
        return self._Enc(ids)


def test_format_example_reading():
    msgs = format_example(ROW, "SYS_R", "SYS_F")
    assert len(msgs) == 3
    assert msgs[0]["content"] == "SYS_R"
    assert "The Fool" in msgs[1]["content"]
    assert "The Hierophant (reversed | polarity: delayed)" in msgs[1]["content"]
    assert "quá khứ" in msgs[1]["content"]
    assert msgs[2]["content"].startswith("Lá The Fool")


def test_format_example_task_variants():
    assert format_example(SAFETY_ROW, "SYS_R", "SYS_F")[0]["content"] == "SYS_F"
    c = format_example(CORRECTION_ROW, "SYS_R", "SYS_F")
    assert "KHẲNG ĐỊNH SAI" in c[1]["content"]


def test_build_tokenized_completions_only():
    tok = FakeTok()
    msgs = format_example(ROW, "SYS", "SYS")
    out = build_tokenized([msgs], tok, seq_len=64)[0]
    ids = out["input_ids"]
    labels = out["labels"]
    assert all(l == -100 for l in labels[:7])
    # completion span unmasked
    assert any(l != -100 for l in labels)
    # padding masked
    assert labels[-1] == -100


def test_build_tokenized_truncation():
    tok = FakeTok()
    msgs = format_example(ROW, "SYS", "SYS")
    out = build_tokenized([msgs], tok, seq_len=8)
    assert len(out[0]["input_ids"]) == 8


def test_load_rows_deterministic_shuffle_and_cap():
    import tempfile, os

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rows.jsonl"
        rows = [{**ROW, "question_vi": f"q{i}"} for i in range(50)]
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                     encoding="utf-8")
        a = load_rows([str(p)], max_examples=10, seed=42)
        b = load_rows([str(p)], max_examples=10, seed=42)
        assert a == b and len(a) == 10
        full = load_rows([str(p)], seed=42)
        assert len(full) == 50


def test_pick_dtype_cpu():
    # on this box (P106-100 = CC 6.1) must NOT be bf16
    d = pick_dtype()
    assert d in ("no", "fp16", "bf16")
    import torch

    if not torch.cuda.is_available():
        assert d == "no"


def test_tripwire_jaccard_and_threshold():
    assert OrientationTripwire._jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert OrientationTripwire._jaccard(set(), {"a"}) == 0.0
    j = OrientationTripwire._jaccard({"a", "b", "c"}, {"b", "c", "d"})
    assert abs(j - 0.5) < 1e-9
