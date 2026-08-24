"""Serving path (plan W7.2/W7.3/W7.4): FastAPI wrapper around llama-server.

Flow per /reading request:
  1. draw (or accept client draw) via tfvn.tools.Deck — deterministic
  2. assemble byte-stable prefix (tfvn.reading)
  3. crisis gate: validator-owned routing BEFORE the model (policy/safety.md §4)
  4. generate via llama-server
  5. deterministic validators (containment, orientation, keyword collision,
     positions) -> on failure: ONE constrained regeneration with the failure
     reason appended; on second failure return original output with
     validation_warning=true (never silent truncation, never an error)

Run: uvicorn tfvn.serve:app --port 8078   (see scripts/serve.sh)
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from policy.crisis_routing import route_crisis  # noqa: E402
from tfvn.reading import N_CTX, assemble_prompt, truncate_history  # noqa: E402
from tfvn.tools import ClarificationManager, Deck, ToolBox  # noqa: E402
from tfvn.validators import (  # noqa: E402
    load_whitelist,
    validate_card_name_containment,
    validate_keyword_collision,
    validate_orientation_consistency,
)

LLAMA_SERVER = os.environ.get("TAROT_LLAMA_SERVER", "http://127.0.0.1:8079")

CRISIS_MARKERS = re.compile(
    r"(tự tử|suýt chết|kết thúc cuộc đời|không muốn sống|tự làm hại bản thân|"
    r"làm hại người khác|chết đi)", re.I)

MAX_REGENERATIONS = 1  # plan W7.2: exactly one constrained retry

app = FastAPI(title="vn-tarot", version="0.1.0")
_deck = Deck()
_toolbox = ToolBox()
_whitelist = load_whitelist()
_kb_rows = None


def _kb():
    global _kb_rows
    if _kb_rows is None:
        from tfvn.serialise import read_jsonl

        _kb_rows = read_jsonl(ROOT / "kb/cards.jsonl")
    return _kb_rows


class ReadingRequest(BaseModel):
    question_vi: str
    seed: int = 42
    n_cards: int = 3


def _llama_generate(prompt: str, *, max_tokens: int = 700,
                    temperature: float = 0.7) -> str:
    body = {"prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature}
    req = urllib.request.Request(
        LLAMA_SERVER.rstrip("/") + "/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("content", "")


def _validate(output: str, draw: List[dict],
              positions: List[str]) -> Dict[str, Any]:
    failures: List[str] = []
    required = [d["name_en"] for d in draw]
    c = validate_card_name_containment(output, _whitelist, required_names=required)
    if c["missing_required"]:
        failures.append(f"unmentioned_drawn_cards={c['missing_required']}")
    if c["not_in_whitelist"] or c["hallucinated_like"]:
        failures.append(
            f"hallucinated={c['not_in_whitelist'] + c['hallucinated_like']}")
    low = output.lower()
    missing_pos = [p for p in positions if p and p.lower() not in low]
    if missing_pos:
        failures.append(f"missing_positions={missing_pos}")
    o = validate_orientation_consistency(output, draw, _kb())
    if not o["ok"]:
        failures.append(f"orientation={o['violations']}")
    k = validate_keyword_collision(output, draw, _kb())
    if not k["ok"]:
        failures.append(f"collisions={k['collisions']}")
    return {"ok": not failures, "failures": failures}


def generate_validated(prompt: str, draw: List[dict],
                       positions: List[str]) -> Dict[str, Any]:
    """generate -> validators -> ONE constrained regeneration -> warning flag."""
    output = _llama_generate(prompt)
    check = _validate(output, draw, positions)
    if check["ok"]:
        return {"output": output, "validation_warning": False,
                "regenerated": False, "failures": []}

    constraint = (
        prompt + "\n\nRÀNG BUỘC (đọc lại cho đúng): "
        + "; ".join(check["failures"])
        + ". CHỈ nói về các lá đã rút và đúng trạng thái.")
    output2 = _llama_generate(constraint)
    check2 = _validate(output2, draw, positions)
    if check2["ok"]:
        return {"output": output2, "validation_warning": False,
                "regenerated": True, "failures": []}
    # both attempts failed -> ship the ORIGINAL with an explicit warning flag
    return {"output": output, "validation_warning": True,
            "regenerated": True, "failures": check2["failures"]}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "deck_size": len(_deck.names),
            "llama_server": LLAMA_SERVER, "n_ctx": N_CTX}


@app.post("/tools/execute")
def execute_tool(name: str, args: Dict[str, Any] = {}) -> dict:
    return _toolbox.execute(name, args)


def _crisis_gate(question_vi: str) -> Optional[dict]:
    """Validator-owned routing (policy §4): the model never owns this decision."""
    if not CRISIS_MARKERS.search(question_vi or ""):
        return None
    now = datetime.now().astimezone()
    decision = route_crisis(now.replace(tzinfo=None))
    message = decision.fallback_message_vi if \
        decision.routing_mode != "primary_open" else (
        "Bạn đang gặp khủng hoảng, và điều đó rất quan trọng. "
        f"Hãy gọi đường dây nóng Ngày mai ({decision.primary_line_phone}), "
        "mở cửa 13:00–20:30 từ Thứ Tư đến Chủ Nhật. "
        "Trong trường hợp khẩn cấp, gọi ngay 115.")
    return {"crisis_route": True,
            "routing_mode": decision.routing_mode,
            "message_vi": message}


@app.post("/reading")
def reading(req: ReadingRequest) -> dict:
    crisis = _crisis_gate(req.question_vi)
    if crisis is not None:
        return {**crisis, "reading_vi": crisis["message_vi"],
                "stop_reading": True}

    if ClarificationManager.needs_clarification(req.question_vi, req.n_cards):
        return {"clarification": ClarificationManager.clarify(),
                "stop_reading": True}

    draw = _toolbox.execute("draw_cards",
                            {"n": min(max(1, req.n_cards), 10),
                             "seed": req.seed})
    cards = draw["cards"]
    positions = [f"vị trí {i + 1}" for i in range(len(cards))]

    prompt = assemble_prompt(req.question_vi, cards, positions)
    result = generate_validated(prompt, cards, positions)
    return {
        "question_vi": req.question_vi,
        "seed": req.seed,
        "draw": cards,
        "positions": positions,
        "reading_vi": result["output"],
        "validation_warning": result["validation_warning"],
        "validator_failures": result["failures"],
        "regenerated": result["regenerated"],
    }
