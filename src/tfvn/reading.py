"""RAG prompt assembly with prompt-cache discipline (plan W7.1).

Assembly order is fixed and each block is byte-stable:

  1. SYSTEM    — constant identity + rules text (never varies)
  2. CARD BLOCK— one canonical-JSON line per drawn card, rendered ONCE per
                 reading from kb/compact_cards.jsonl via the canonical
                 serialiser (sorted keys, compact separators)
  3. POSITIONS — spread glosses for the drawn positions only
  4. QUESTION  — volatile per turn

Golden-file guarantee: same draw -> byte-identical prefix across calls,
processes and Python versions (dict ordering never enters the output because
the canonical serialiser sorts keys).

n_ctx = 4096 (documented plan value). Multi-turn truncation policy: when the
accumulated history exceeds budget after the stable prefix, drop OLDEST
user/assistant exchanges first; the system+card+position prefix is never
truncated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import dumps_canonical, read_jsonl  # noqa: E402

N_CTX = 4096

SYSTEM_BLOCK = (
    "Bạn là chuyên gia đọc bài Tarot bằng tiếng Việt tự nhiên.\n"
    "QUY TẮC:\n"
    "- Chỉ nói về các lá đã rút; không bịa lá khác.\n"
    "- Phân biệt rõ nghĩa XUÔI và NGƯỢC.\n"
    "- Tên lá bài giữ tiếng Anh nguyên văn.\n"
    "- Câu hỏi thuộc lĩnh vực y khoa/pháp lý/khủng hoảng thì từ chối và "
    "giới thiệu chuyên gia hoặc đường dây nóng phù hợp."
)

_SEPARATOR = "\n---\n"


def load_compact_index(path: Optional[Path] = None) -> Dict[tuple, dict]:
    path = path or ROOT / "kb/compact_cards.jsonl"
    return {(int(r["card_id"]), r["orientation"]): r for r in read_jsonl(path)}


def card_block(draw: Sequence[dict], compact: Optional[Dict[tuple, dict]] = None) -> str:
    """Canonical JSON lines for the drawn cards, in draw order."""
    idx = compact if compact is not None else load_compact_index()
    lines = []
    for d in draw:
        key = (int(d["card_id"]), d["orientation"])
        row = idx.get(key)
        if row is None:
            raise KeyError(f"compact KB missing {key}")
        lines.append(dumps_canonical(row))
    return "BÀI ĐÃ RÚT:\n" + "\n".join(lines)


def position_block(positions: Sequence[str]) -> str:
    return "CÁC VỊ TRÍ:\n" + "\n".join(f"- {i + 1}. {p}"
                                       for i, p in enumerate(positions))


def assemble_prefix(draw: Sequence[dict], positions: Sequence[str],
                    compact: Optional[Dict[tuple, dict]] = None) -> str:
    """The cache-stable prefix: system + card block + positions."""
    return _SEPARATOR.join([
        "SYSTEM:\n" + SYSTEM_BLOCK,
        card_block(draw, compact),
        position_block(positions),
    ])


def assemble_prompt(question: str, draw: Sequence[dict], positions: Sequence[str],
                    compact: Optional[Dict[tuple, dict]] = None) -> str:
    """Full prompt for one turn: prefix + user question."""
    return assemble_prefix(draw, positions, compact) + \
        _SEPARATOR + "CÂU HỎI: " + question.strip()


def truncate_history(history: Sequence[dict], token_budget: int,
                     tokenizer_len=lambda s: len(s) // 3) -> List[dict]:
    """Drop oldest exchanges first until history fits the remaining budget;
    never touch the stable prefix (caller accounts for it separately)."""
    kept = list(history)
    while kept and sum(tokenizer_len(m["content"]) for m in kept) > token_budget:
        kept.pop(0)
    return kept
