"""Streaming multi-turn reading pipeline with call-trace logging.

Extends the W7 serving path (``tfvn.serve``) with what the local webapp's
Readings view needs:

* sessions with a FIXED draw — one draw per conversation, so the stable
  prefix (system + card block + positions) stays byte-identical across
  follow-up turns and llama-server's prompt cache keeps hitting (W7.1);
* token streaming end-to-end: llama-server ``stream:true`` -> async token
  deltas -> SSE frames (framing happens in the router);
* deterministic validators on each COMPLETED stream with exactly ONE
  constrained regeneration (W7.2 discipline, same failure-reason text);
* call-trace logging: every pipeline step is both yielded as an event AND
  appended as a canonical-JSON line to ``logs/readings/<session_id>.jsonl``.

Safety-critical pieces (crisis markers/gate, validators, retry constraint)
are IMPORTED from ``tfvn.serve`` rather than duplicated — one source of
truth; this module adds streaming/session orchestration around them.

History/truncation policy (W7.1): ``n_ctx=4096``; when accumulated history
exceeds the post-prefix budget the OLDEST exchanges drop first via
``tfvn.reading.truncate_history``; the stable prefix is never truncated.

In-memory session state requires uvicorn workers=1 without --reload — same
discipline as ``tfvn.webapp.runs``.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# tfvn.serve imports policy.crisis_routing — resolve regardless of launch cwd
# (uvicorn's own entry relies on cwd being the repo root).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from tfvn.reading import N_CTX, assemble_messages, truncate_history  # noqa: E402
from tfvn.serialise import dumps_canonical  # noqa: E402

# Single source of truth for safety-critical behaviour (see module docstring).
from tfvn.serve import (  # noqa: E402  (private-by-location, shared by contract)
    MAX_REGENERATIONS,
    _crisis_gate,
    _validate,
)

MAX_TOKENS = 700          # matches tfvn.serve._llama_generate default
BUDGET_MARGIN = 256       # headroom for the next user turn + KV overhead
SESSION_CAP = 100         # bound memory; evict oldest beyond this

LOG_DIR = ROOT / "logs" / "readings"

_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=30.0)


def llama_server() -> str:
    """Read per-call so .env loaded at webapp lifespan startup wins."""
    return os.environ.get("TAROT_LLAMA_SERVER", "http://127.0.0.1:8079").rstrip("/")


def _client() -> httpx.AsyncClient:
    """Client factory — tests swap this for a MockTransport-backed client."""
    return httpx.AsyncClient(timeout=_TIMEOUT)


def probe_backend(timeout: float = 2.0) -> Dict[str, Any]:
    """Sync health probe of llama-server for the UI banner."""
    try:
        with httpx.Client(timeout=timeout) as cli:
            resp = cli.get(llama_server() + "/health")
            ok = resp.status_code == 200
        return {"ok": ok, "llama_server": llama_server()}
    except Exception as exc:  # connection refused / timeout — banner material
        return {"ok": False, "llama_server": llama_server(),
                "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Sessions — fixed draw per conversation (prompt-cache discipline)
# --------------------------------------------------------------------------- #


@dataclass
class ReadingSession:
    session_id: str
    seed: int
    n_cards: int
    created_at: float = field(default_factory=time.time)
    draw: Optional[List[Dict[str, Any]]] = None      # set on first real turn
    positions: List[str] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)  # full chat
    turns: int = 0


_SESSIONS: "deque[ReadingSession]" = deque(maxlen=SESSION_CAP)


def new_session(seed: Optional[int] = None, n_cards: int = 3) -> ReadingSession:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    session = ReadingSession(
        session_id=uuid.uuid4().hex[:12],
        seed=int(seed),
        n_cards=max(1, min(int(n_cards), 10)),
    )
    _SESSIONS.append(session)
    return session


def get_session(session_id: str) -> Optional[ReadingSession]:
    for s in _SESSIONS:
        if s.session_id == session_id:
            return s
    return None


def delete_session(session_id: str) -> bool:
    for i, s in enumerate(_SESSIONS):
        if s.session_id == session_id:
            del _SESSIONS[i]
            return True
    return False


# --------------------------------------------------------------------------- #
# Call-trace logging (JSONL, canonical bytes)
# --------------------------------------------------------------------------- #


def _trace(log_path: Path, event: Dict[str, Any]) -> None:
    line = dumps_canonical({"ts": datetime.now().astimezone().isoformat(),
                            **event})
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # tracing must never break a reading


# --------------------------------------------------------------------------- #
# Streaming generation
# --------------------------------------------------------------------------- #


async def _stream_llama(messages: List[Dict[str, str]]
                        ) -> AsyncIterator[str]:
    """Yield content deltas from llama-server chat completions (stream:true).
    Raises RuntimeError with the HTTP status on a failed request."""
    body = {"messages": messages, "max_tokens": MAX_TOKENS,
            "temperature": 0.7, "stream": True}
    async with _client() as cli:
        async with cli.stream("POST",
                              llama_server() + "/v1/chat/completions",
                              json=body) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", "replace")[:200]
                raise RuntimeError(f"llama-server {resp.status_code}: {text}")
            async for raw in resp.aiter_lines():
                if not raw.startswith("data:"):
                    continue
                payload = raw[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece


_RETRY_CONSTRAINT = (
    "RÀNG BUỘC (đọc lại cho đúng): {failures}. "
    "CHỈ nói về các lá đã rút và đúng trạng thái.")


def _retry_messages(messages: List[Dict[str, str]],
                    failures: List[str]) -> List[Dict[str, str]]:
    constraint = _RETRY_CONSTRAINT.format(failures="; ".join(failures))
    return messages[:-1] + [
        {"role": "user",
         "content": messages[-1]["content"] + "\n" + constraint}]


# --------------------------------------------------------------------------- #
# History budgeting (W7.1 truncation policy)
# --------------------------------------------------------------------------- #


def budgeted_messages(session: ReadingSession,
                      ) -> tuple[List[Dict[str, str]], int]:
    """History-carrying messages WITHOUT the new user turn: the stable
    prefix ``messages[:2]`` (system + first user turn) plus older exchanges
    trimmed OLDEST-first to fit the post-prefix budget.
    Returns (messages, dropped_count)."""
    if len(session.messages) < 3:
        return list(session.messages), 0
    prefix = session.messages[:2]
    prefix_est = sum(len(m["content"]) // 3 for m in prefix)
    budget = max(N_CTX - MAX_TOKENS - BUDGET_MARGIN - prefix_est, 1)
    kept = truncate_history(session.messages[2:], budget)
    dropped = (len(session.messages) - 2) - len(kept)
    return prefix + kept, dropped


# --------------------------------------------------------------------------- #
# The turn pipeline
# --------------------------------------------------------------------------- #


async def stream_turn(session: ReadingSession, content: str,
                      *, log_dir: Optional[Path] = None,
                      ) -> AsyncIterator[Dict[str, Any]]:
    """One conversation turn as an event stream.

    Event types: step | stop | tokens | validate | regen | warning | done.
    Token deltas are traced compactly (chars only) so the JSONL trace stays
    small; every other event is stored whole. History is committed ONLY on
    'done' — an interrupted stream leaves the session unchanged, so a client
    may safely resend the same turn.
    """
    started = time.perf_counter()
    log_path = (log_dir or LOG_DIR) / f"{session.session_id}.jsonl"

    def emit(event: Dict[str, Any]) -> Dict[str, Any]:
        _trace(log_path, {"session_id": session.session_id, **event})
        return event

    content = (content or "").strip()
    if not content:
        yield emit({"type": "stop", "reason": "empty_question",
                    "message_vi": "Bạn chưa nhập câu hỏi."})
        return

    # 1. crisis gate — validator-owned, BEFORE anything else, every turn.
    crisis = _crisis_gate(content)
    yield emit({"type": "step", "stage": "crisis_gate",
                "routed": crisis is not None})
    if crisis is not None:
        yield emit({"type": "stop", "reason": "crisis",
                    "routing_mode": crisis["routing_mode"],
                    "message_vi": crisis["message_vi"]})
        return

    # 2. clarification — only before any cards are drawn (ONE question).
    if session.draw is None:
        from tfvn.tools import ClarificationManager

        if ClarificationManager.needs_clarification(content, session.n_cards):
            message = ClarificationManager.clarify()
            yield emit({"type": "step", "stage": "clarification"})
            yield emit({"type": "stop", "reason": "clarification",
                        "message_vi": message})
            return

    # 3. fixed draw — once per session, never again (byte-stable prefix).
    if session.draw is None:
        from tfvn.tools import Deck

        deck = Deck()
        session.draw = deck.draw(session.n_cards, session.seed)
        session.positions = [f"vị trí {i + 1}"
                             for i in range(len(session.draw))]
        yield emit({"type": "step", "stage": "draw",
                    "cards": session.draw, "seed": session.seed})

    # 4. assemble messages (stable prefix + truncated history + new turn).
    if session.messages:
        msgs, dropped = budgeted_messages(session)
        yield emit({"type": "step", "stage": "context",
                    "history_messages": len(msgs) - 2, "dropped": dropped})
        msgs.append({"role": "user", "content": content})
    else:
        msgs = assemble_messages(content, session.draw, session.positions)
        yield emit({"type": "step", "stage": "context",
                    "history_messages": 0, "dropped": 0})

    # 5. generate -> validate -> ONE constrained regen (serve.py semantics).
    output: Optional[str] = None
    first_output = ""
    regenerated = False
    warning = False
    failures: List[str] = []
    gen_started = time.perf_counter()
    n_deltas = 0

    for attempt in range(1 + MAX_REGENERATIONS):
        parts: List[str] = []
        n_deltas = 0
        turn_msgs = msgs if attempt == 0 else _retry_messages(msgs, failures)
        gen_started = time.perf_counter()
        try:
            async for piece in _stream_llama(turn_msgs):
                parts.append(piece)
                n_deltas += 1
                _trace(log_path, {"session_id": session.session_id,
                                  "type": "tokens", "chars": len(piece)})
                yield {"type": "tokens", "text": piece}
        except RuntimeError as exc:
            yield emit({"type": "stop", "reason": "backend_error",
                        "message_vi": "Không kết nối được llama-server.",
                        "error": str(exc)})
            return
        candidate = "".join(parts)
        if attempt == 0:
            first_output = candidate
        elapsed_s = time.perf_counter() - gen_started
        check = _validate(candidate, session.draw, session.positions)
        failures = check["failures"]
        yield emit({"type": "validate", "attempt": attempt,
                    "ok": check["ok"], "failures": failures,
                    "elapsed_ms": int(elapsed_s * 1000),
                    "tok_s_approx": round(n_deltas / elapsed_s, 1)
                    if elapsed_s > 0 else None})
        if check["ok"]:
            output = candidate
            regenerated = attempt > 0
            break
        if attempt < MAX_REGENERATIONS:
            yield emit({"type": "regen", "failures": failures})

    if output is None:
        # Both attempts failed validation — ship the ORIGINAL attempt with
        # the explicit warning flag (serve.py semantics: never silent, never
        # an error). A regeneration WAS attempted.
        output = first_output
        warning = True
        regenerated = True

    # 6. commit — only now does the session remember anything. msgs already
    # ends with this turn's user content; the stable prefix rides along so a
    # follow-up turn rebuilds a byte-identical leading sequence.
    session.messages = msgs + [{"role": "assistant", "content": output}]
    session.turns += 1
    total_ms = int((time.perf_counter() - started) * 1000)
    yield emit({
        "type": "done", "turn": session.turns,
        "validation_warning": warning, "regenerated": regenerated,
        "validator_failures": failures if warning else [],
        "draw": session.draw, "positions": session.positions,
        "elapsed_ms": total_ms,
    })
