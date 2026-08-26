"""Readings router — multi-turn streaming tarot sessions for the SPA.

Endpoints (all under ``/api/readings``):
  GET    /backend              health probe of llama-server (banner source)
  POST   /session              create an empty session {n_cards, seed?}
  GET    /{session_id}         session state (draw once made, turns)
  DELETE /{session_id}         drop a session ("New reading" redraws)
  POST   /{session_id}/turn    one conversation turn as text/event-stream

SSE frame per pipeline event: ``event: <type>\\ndata: <canonical json>\\n\\n``
with types step | stop | tokens | validate | regen | warning | done.

The turn endpoint streams directly from :func:`tfvn.reading_stream.stream_turn`;
an interrupted connection cancels the generator BEFORE any history commit, so
the client may simply resend the same content (graceful reconnect). Session
state is in-memory — uvicorn workers=1, no --reload (same discipline as
``tfvn.webapp.runs``).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..reading_stream import (
    delete_session,
    get_session,
    new_session,
    probe_backend,
    stream_turn,
)
from ..serialise import dumps_canonical

router = APIRouter(prefix="/api/readings")


class SessionRequest(BaseModel):
    n_cards: int = 3
    seed: Optional[int] = None


class TurnRequest(BaseModel):
    content: str


def _sse(event: dict[str, Any]) -> bytes:
    return (
        f"event: {event['type']}\n"
        f"data: {dumps_canonical(event)}\n\n"
    ).encode("utf-8")


def _require(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown session {session_id}")
    return session


@router.get("/backend")
def backend_probe() -> dict[str, Any]:
    """Registered BEFORE the ``/{session_id}`` route so it never matches."""
    return probe_backend()


@router.post("/session")
def create_session(req: SessionRequest) -> dict[str, Any]:
    s = new_session(seed=req.seed, n_cards=req.n_cards)
    return {"session_id": s.session_id, "seed": s.seed,
            "n_cards": s.n_cards}


@router.get("/{session_id}")
def session_state(session_id: str) -> dict[str, Any]:
    s = _require(session_id)
    return {"session_id": s.session_id, "seed": s.seed,
            "n_cards": s.n_cards, "draw": s.draw,
            "positions": s.positions, "turns": s.turns}


@router.delete("/{session_id}")
def drop_session(session_id: str) -> dict[str, Any]:
    _require(session_id)
    delete_session(session_id)
    return {"deleted": session_id}


@router.post("/{session_id}/turn")
async def turn(session_id: str, req: TurnRequest) -> StreamingResponse:
    session = _require(session_id)

    async def gen():
        async for event in stream_turn(session, req.content):
            yield _sse(event)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
