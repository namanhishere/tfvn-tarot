"""Tests for the readings pipeline (tfvn.reading_stream) and its webapp
router (tfvn.webapp.readings).

llama-server is replaced by an ``httpx.MockTransport`` handler installed via
the ``fake_llama`` fixture — no network, no GPU. The handler is a plain
function each test can swap, plus a ``calls`` counter for asserting exactly
how many model requests a turn made.

SSE frames from the router are parsed back into event dicts by
:func:`parse_sse` so assertions read like the pipeline contract:
step | stop | tokens | validate | regen | warning | done.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tfvn import reading_stream as rs
from tfvn.webapp.readings import router


# --------------------------------------------------------------------------- #
# Fake llama-server
# --------------------------------------------------------------------------- #

GOOD_REPLY = (
    "Bài của bạn: vị trí {i} có lá {name}. "
    "Câu chuyện cần sự kiên nhẫn và lòng tin."
)

STATE: dict[str, Any] = {"handler": None, "calls": 0}


def _default_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body.get("stream") is True, "pipeline must request stream:true"
    user = body["messages"][-1]["content"]
    names = [json.loads(l)["name_en"]
             for l in user.splitlines() if l.startswith('{"card_id"')]
    text = " ".join(
        GOOD_REPLY.format(i=i + 1, name=n) for i, n in enumerate(names))
    half = len(text) // 2
    return httpx.Response(200, content=_sse_text([text[:half], text[half:]]))


def _sse_text(chunks: list[str]) -> str:
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": c}}]})
             for c in chunks]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


@pytest.fixture()
def fake_llama(monkeypatch, tmp_path):
    """Route reading_stream's llama traffic to MockTransport; isolate traces."""
    STATE["handler"] = _default_handler
    STATE["calls"] = 0
    monkeypatch.setattr(rs, "_log_dir", lambda: tmp_path / "traces")

    def counting_handler(request):
        STATE["calls"] += 1
        return STATE["handler"](request)

    monkeypatch.setattr(
        rs, "_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(counting_handler)))
    return STATE


def parse_sse(lines: list[str]) -> list[dict[str, Any]]:
    events = []
    data = None
    for line in lines:
        if line.startswith("data: "):
            data = json.loads(line[len("data: "):])
        elif line == "" and data is not None:
            events.append(data)
            data = None
    return events


def make_client():
    from tests.webapp.conftest import make_client as _mc

    return _mc(router)


def post_turn(client, sid: str, content: str) -> list[dict[str, Any]]:
    with client.stream("POST", f"/api/readings/{sid}/turn",
                       json={"content": content}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return parse_sse([ln.rstrip("\n") for ln in resp.iter_lines()])


# --------------------------------------------------------------------------- #
# Session lifecycle (pure router)
# --------------------------------------------------------------------------- #


def test_session_lifecycle():
    client = make_client()
    r = client.post("/api/readings/session",
                    json={"n_cards": 3, "seed": 42})
    assert r.status_code == 200
    body = r.json()
    assert body["n_cards"] == 3 and body["seed"] == 42 and body["session_id"]

    state = client.get(f"/api/readings/{body['session_id']}").json()
    assert state["draw"] is None and state["turns"] == 0

    assert client.delete(
        f"/api/readings/{body['session_id']}").json()["deleted"] \
        == body["session_id"]
    assert client.get(f"/api/readings/{body['session_id']}").status_code == 404
    assert client.delete(
        f"/api/readings/{body['session_id']}").status_code == 404


def test_session_clamps_n_cards():
    client = make_client()
    assert client.post("/api/readings/session",
                       json={"n_cards": 99}).json()["n_cards"] == 10
    assert client.post("/api/readings/session",
                       json={"n_cards": 0}).json()["n_cards"] == 1


def test_turn_unknown_session_404():
    client = make_client()
    r = client.post("/api/readings/deadbeef/turn",
                    json={"content": "Tình yêu?"})
    assert r.status_code == 404


def test_backend_probe_endpoint(monkeypatch):
    client = make_client()
    import tfvn.webapp.readings as rr

    monkeypatch.setattr(rr, "probe_backend",
                        lambda timeout=2.0: {"ok": False,
                                             "error": "ConnectError"})
    body = client.get("/api/readings/backend").json()
    assert body == {"ok": False, "error": "ConnectError"}


# --------------------------------------------------------------------------- #
# Turn pipeline over SSE
# --------------------------------------------------------------------------- #


def test_first_turn_streams_draw_validates_and_commits(fake_llama):
    client = make_client()
    sid = client.post("/api/readings/session",
                      json={"n_cards": 3, "seed": 42}).json()["session_id"]

    events = post_turn(client, sid, "Tình yêu của tôi sẽ thế nào?")
    kinds = [e["type"] for e in events]

    assert kinds[0] == "step" and events[0]["stage"] == "crisis_gate"
    draw_steps = [e for e in events
                  if e["type"] == "step" and e.get("stage") == "draw"]
    assert len(draw_steps) == 1 and len(draw_steps[0]["cards"]) == 3
    assert sum(1 for k in kinds if k == "tokens") >= 2
    validates = [e for e in events if e["type"] == "validate"]
    assert len(validates) == 1 and validates[0]["ok"] is True
    done = events[-1]
    assert done["type"] == "done" and done["validation_warning"] is False

    state = client.get(f"/api/readings/{sid}").json()
    assert state["turns"] == 1 and len(state["draw"]) == 3


def test_fixed_draw_and_prefix_across_followup(fake_llama):
    client = make_client()
    sid = client.post("/api/readings/session",
                      json={"n_cards": 1, "seed": 7}).json()["session_id"]

    ev1 = post_turn(client, sid, "Tình yêu của tôi sẽ thế nào?")
    ev2 = post_turn(client, sid, "Còn công việc thì sao?")

    draw1 = [e for e in ev1 if e.get("stage") == "draw"][0]
    assert not any(e.get("stage") == "draw" for e in ev2), \
        "follow-up must reuse the fixed draw"
    ctx = [e for e in ev2 if e.get("stage") == "context"][0]
    assert ctx["history_messages"] >= 1 and ctx["dropped"] == 0
    assert client.get(f"/api/readings/{sid}").json()["turns"] == 2

    # trace proves the same session file accumulated both turns' steps
    trace = rs._log_dir() / f"{sid}.jsonl"
    lines = trace.read_text().splitlines()
    sids = {json.loads(l)["session_id"] for l in lines}
    assert sids == {sid}


def test_clarification_before_any_draw(fake_llama):
    client = make_client()
    sid = client.post("/api/readings/session",
                      json={"n_cards": 3}).json()["session_id"]

    events = post_turn(client, sid, "bói đi")
    stop = events[-1]
    assert stop["type"] == "stop" and stop["reason"] == "clarification"
    assert "lĩnh vực" in stop["message_vi"]
    assert fake_llama["calls"] == 0
    assert client.get(f"/api/readings/{sid}").json()["draw"] is None


def test_crisis_routes_without_model_call(fake_llama):
    client = make_client()
    sid = client.post("/api/readings/session",
                      json={"n_cards": 1}).json()["session_id"]

    events = post_turn(client, sid, "Tôi không muốn sống nữa")
    stop = events[-1]
    assert stop["type"] == "stop" and stop["reason"] == "crisis"
    assert stop["message_vi"]          # hotline or static fallback present
    assert fake_llama["calls"] == 0


def test_validator_failure_regen_then_warning(fake_llama, monkeypatch):
    def bad_handler(request):
        return httpx.Response(200, content=_sse_text(["Trời hôm nay đẹp."]))

    fake_llama["handler"] = bad_handler
    client = make_client()
    sid = client.post("/api/readings/session",
                      json={"n_cards": 1, "seed": 99}).json()["session_id"]

    events = post_turn(client, sid, "Tình yêu của tôi sẽ thế nào?")
    validates = [e for e in events if e["type"] == "validate"]
    assert [v["attempt"] for v in validates] == [0, 1]
    assert all(not v["ok"] for v in validates)
    assert any(e["type"] == "regen" for e in events)
    done = events[-1]
    assert done["validation_warning"] is True
    assert done["regenerated"] is True
    assert fake_llama["calls"] == 2   # original + ONE constrained retry


def test_backend_error_leaves_session_clean(fake_llama):
    def broken(request):
        return httpx.Response(500, text="boom")

    fake_llama["handler"] = broken
    client = make_client()
    sid = client.post("/api/readings/session",
                      json={"n_cards": 1}).json()["session_id"]

    events = post_turn(client, sid, "Tình yêu của tôi sẽ thế nào?")
    stop = events[-1]
    assert stop["type"] == "stop" and stop["reason"] == "backend_error"

    state = client.get(f"/api/readings/{sid}").json()
    assert state["turns"] == 0        # nothing committed -> safe to retry


def test_empty_content_stops_immediately(fake_llama):
    client = make_client()
    sid = client.post("/api/readings/session",
                      json={"n_cards": 1}).json()["session_id"]
    events = post_turn(client, sid, "   ")
    assert events[-1] == {"type": "stop", "reason": "empty_question",
                          "message_vi": "Bạn chưa nhập câu hỏi."}
    assert fake_llama["calls"] == 0
