"""FastAPI application for the local dataset viewer.

A.1 skeleton: health endpoint only. A.2–A.5 modules get wired into routes by
todo A.6 (server wiring + static mount + lifespan warm-up).
"""

from __future__ import annotations

from fastapi import FastAPI

app: FastAPI = FastAPI(title="tfvn dataset viewer")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe — must not depend on .env or any data file."""
    return {"status": "ok"}
