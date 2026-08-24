"""Tool-call system (plan scope: deck fold, card meanings, clarification).

Deterministic deck operations + a tool registry the serving path exposes to
the model. Everything is seed-reproducible; no wall-clock, no hidden state.

Tools:
  shuffle_deck   — deterministic Fisher-Yates over the 78-card spine
  draw_cards     — draw n cards with orientation from a seeded shuffle
  get_card_meaning — compact KB row for (card_id, orientation)
  list_spreads   — available spreads with sizes
  ask_clarification — structured prompt when the request is ambiguous
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


DECK_SIZE = 78
REVERSED_P = 0.5

from tfvn.serialise import read_jsonl  # noqa: E402


class Deck:
    """Seeded deck over the canonical 0–77 spine."""
    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or ROOT / "kb/cards.jsonl"
        rows = read_jsonl(self.kb_path)
        self.names: Dict[int, str] = {}
        for r in rows:
            if r.get("orientation") == "upright":
                self.names[int(r["card_id"])] = r["name_en"]
        assert len(self.names) == DECK_SIZE, f"spine incomplete: {len(self.names)}"

    def shuffle(self, seed: int) -> List[int]:
        rng = random.Random(seed)
        order = list(range(DECK_SIZE))
        rng.shuffle(order)
        return order

    def draw(self, n: int, seed: int) -> List[Dict[str, Any]]:
        """First n cards of the seeded shuffle, each flipped to reversed with p=0.5."""
        assert 1 <= n <= DECK_SIZE
        rng = random.Random(seed * 1_000_003 + 7)  # orientation stream
        order = self.shuffle(seed)
        out = []
        for cid in order[:n]:
            reversed_ = rng.random() < REVERSED_P
            out.append({"card_id": cid, "name_en": self.names[cid],
                        "orientation": "reversed" if reversed_ else "upright"})
        return out


def load_compact(path: Optional[Path] = None) -> Dict[tuple, dict]:
    path = path or ROOT / "kb/compact_cards.jsonl"
    return {(int(r["card_id"]), r["orientation"]): r for r in read_jsonl(path)}


# --------------------------------------------------------------- registry --

TOOL_SPECS = [
    {
        "name": "draw_cards",
        "description": "Rút ngẫu nhiên n lá bài từ bộ Tarot 78 lá (deterministic theo seed).",
        "parameters": {"n": "int (1-10)", "seed": "int", "spread_id": "str (tuỳ chọn)"},
    },
    {
        "name": "get_card_meaning",
        "description": "Lấy nghĩa rút gọn của một lá bài ở trạng thái xuôi hoặc đảo.",
        "parameters": {"card_id": "int (0-77)", "orientation": "'upright' | 'reversed'"},
    },
    {
        "name": "list_spreads",
        "description": "Liệt kê các kiểu trải bài và số lá tương ứng.",
        "parameters": {},
    },
]


class ToolBox:
    """Executes model-emitted tool calls against local artifacts."""

    def __init__(self, kb_path: Optional[Path] = None,
                 compact_path: Optional[Path] = None,
                 spreads_path: Optional[Path] = None):
        self.deck = Deck(kb_path)
        self.compact = load_compact(compact_path)
        spreads_rows = read_jsonl(spreads_path or ROOT / "kb/spreads.jsonl")
        self.spreads = [{"spread_id": s["spread_id"], "name_en": s["name_en"],
                         "cards_drawn": s["cards_drawn"]} for s in spreads_rows]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "draw_cards":
            n = int(args.get("n", 3))
            seed = int(args.get("seed", 42))
            drawn = self.deck.draw(n, seed)
            spread_id = args.get("spread_id")
            spread = next((s for s in self.spreads
                           if s["spread_id"] == spread_id), None)
            positions = []
            if spread and spread["cards_drawn"] == n:
                positions = [p for p in read_jsonl(
                    ROOT / "kb/spreads.jsonl")
                    if p["spread_id"] == spread_id][0]["positions"]
            result = {"cards": drawn}
            if positions:
                result["positions"] = [p.get("label_vi") or p.get("label_en")
                                       or p.get("gloss_compact") for p in positions]
            return result
        if name == "get_card_meaning":
            key = (int(args["card_id"]), args["orientation"])
            row = self.compact.get(key)
            if not row:
                return {"error": f"no KB row for {key}"}
            return {"card_id": row["card_id"], "name_en": row["name_en"],
                    "orientation": row["orientation"],
                    "keywords_en": row["keywords_en"],
                    "polarity_axis": row["polarity_axis"]}
        if name == "list_spreads":
            return {"spreads": self.spreads}
        if name == "ask_clarification":
            question = ClarificationManager.clarify(args.get("missing", []))
            return {"clarification": question, "stop_reading": True}
        return {"error": f"unknown tool: {name}"}


class ClarificationManager:
    """Ambiguity gate: a reading request missing topic/scope gets ONE clarifying
    question instead of a guessed reading."""

    REQUIRED_SLOTS = ("topic",)

    AMBIGUOUS_MARKERS = re.compile(
        r"^\s*(bói đi|xem giúp|cho xin lá bài|rút bài thôi|còn gì nữa không)\s*[?.!]*\s*$",
        re.I)

    @classmethod
    def needs_clarification(cls, question: str, n_cards_requested: Optional[int]) -> bool:
        q = (question or "").strip()
        if not q:
            return True
        if cls.AMBIGUOUS_MARKERS.match(q):
            return True
        has_topic = any(k in q.lower() for k in
                        ("tình yêu", "công việc", "tiền", "tài chính", "sức khỏe",
                         "gia đình", "học", "quyết định", "tương lai"))
        return not has_topic

    @staticmethod
    def clarify(missing: Sequence[str] = ()) -> str:
        slots = set(missing) or set(ClarificationManager.REQUIRED_SLOTS)
        parts = []
        if "topic" in slots:
            parts.append("bạn muốn hỏi về lĩnh vực nào (tình yêu, công việc, "
                         "tài chính, sức khỏe hay một quyết định cụ thể)?")
        if "n_cards" in slots:
            parts.append("bạn muốn trải bài 1 lá hay 3 lá?")
        return "Để lời đọc chính xác hơn, " + " Và ".join(parts)


