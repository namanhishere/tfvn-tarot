"""FastAPI application for the local dataset viewer (todo A.6).

Assembles the Wave-A routers — catalog (A.2), stats (A.3), filtering (A.4),
runs (A.5) and reports (gap closure for the B.7 Reports view) — plus:

* ``GET /api/health`` — liveness probe (kept from A.1).
* Lifespan: :func:`load_env` from ``tfvn.llm_client`` so re-run subprocesses
  inherit ``.env`` keys, then warms the stats cache in a background task so
  the first ``/api/stats`` call is served from cache.
* ``StaticFiles(directory=webapp/static)`` mounted at ``/`` as the LAST route
  so ``/api/*`` paths never fall through to the SPA.
* A JSON 404 handler for unknown ``/api/*`` paths.

CORS is intentionally not configured: the SPA is served same-origin by this
app and never calls out.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..llm_client import load_env
from .catalog import router as catalog_router
from .filtering import router as filtering_router
from .readings import router as readings_router
from .reports import router as reports_router
from .runs import router as runs_router
from .stats import router as stats_router

REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = REPO_ROOT / "webapp" / "static"


async def _warm_stats() -> None:
    """Best-effort stats computation so the first /api/stats is cache-served."""
    from .stats import compute_stats

    try:
        await asyncio.to_thread(compute_stats)
    except Exception:
        pass  # warm is opportunistic; a real call recomputes on failure


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_env()
    warm = asyncio.create_task(_warm_stats())
    try:
        yield
    finally:
        warm.cancel()


app = FastAPI(title="tfvn dataset viewer", lifespan=lifespan)

# API routers first — order of registration is route-match order, and the
# static mount at "/" must never shadow them.
app.include_router(catalog_router)
app.include_router(stats_router)
app.include_router(filtering_router)
app.include_router(runs_router)
app.include_router(reports_router)
app.include_router(readings_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe — must not depend on .env or any data file."""
    return {"status": "ok"}


@app.exception_handler(404)
async def not_found_json(_: Request, exc: HTTPException) -> JSONResponse:
    """Unknown paths return JSON ``{"detail": "Not Found"}``, never HTML."""
    return JSONResponse(
        status_code=exc.status_code, content={"detail": "Not Found"}
    )


# Static SPA mount LAST — after every /api/* route so API paths never fall
# through to the file server. html=True serves index.html at "/".
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
