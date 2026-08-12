"""Server assembly tests (A.6 / C.1).

Exercises the real ``server.app`` — health probe, JSON 404 handler and the
static SPA mount. The lifespan's stats warm-up is monkeypatched to a no-op so
boot never computes stats over the real repo.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from tfvn.webapp import server

    async def _no_warm() -> None:
        return None

    monkeypatch.setattr(server, "_warm_stats", _no_warm)
    with TestClient(server.app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_unknown_api_path_returns_json_404(client):
    r = client.get("/api/definitely-not-a-route")
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}
    assert r.headers["content-type"].startswith("application/json")


def test_static_spa_mount_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_api_routers_wired(client):
    """All five Wave-A routers are mounted on the real app."""
    assert client.get("/api/catalog").status_code == 200
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/rows/cards?page=1&page_size=5").status_code == 200
    assert client.get("/api/runs").status_code == 200
    assert client.get("/api/reports").status_code == 200
