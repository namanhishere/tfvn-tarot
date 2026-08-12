"""Filtered row API, text search and JSONL export (todo A.4).

Row browser over the post-Wave-3 artifacts:

* ``GET /api/rows/{dataset_id}`` — paginated, filtered, searchable rows.
* ``GET /api/rows/{dataset_id}/{id_path}`` — single raw record by the
  dataset's primary key (SFT/raw/ifd → ``example_id``; cards/spines →
  ``{card_id}/{orientation}``; ``vn_upright`` → ``card_id``; spreads →
  ``spread_id``; anchor → ``anchor_id``).
* ``GET /api/export/{dataset_id}`` — same filters as the rows endpoint but
  streams every matching row as canonical JSONL (one ``dumps_canonical``
  line per row, chunked via a generator) with ``Content-Disposition`` set.

The dataset registry and per-row predicates live in ``filtering_data`` so this
module stays under the LOC ceiling; the public functions accept ``root`` for
testability (C.1 fixtures) and the ``router`` is mounted by server.py in A.6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Iterator, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..serialise import dumps_canonical
from .filtering_data import DATASETS, iter_matching, iter_rows

REPO_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# Query / response models
# --------------------------------------------------------------------------- #

class RowsParams(BaseModel):
    """Query params shared by the rows and export endpoints.

    Missing params are ignored; invalid values (page < 1, page_size > 200,
    non-int card_id, non-float ifd bounds, tier not core|bulk) → 422.
    """

    model_config = ConfigDict(populate_by_name=True)

    page: int = Field(1, ge=1, description="1-based page")
    page_size: int = Field(50, ge=1, le=200, description="rows per page (max 200)")
    q: Optional[str] = Field(None, description="case-insensitive substring search")
    task_type: Optional[str] = None
    # "register" shadows BaseModel.register → keep the attribute private,
    # expose the query param through the alias.
    register_: Optional[str] = Field(None, alias="register")
    length_band: Optional[str] = None
    querent_context: Optional[str] = None
    spread_id: Optional[str] = None
    card_id: Optional[int] = None
    orientation: Optional[str] = None
    tier: Optional[Literal["core", "bulk"]] = None
    ifd_min: Optional[float] = None
    ifd_max: Optional[float] = None


class RowsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    rows: list[dict[str, Any]]


def _require_dataset(dataset_id: str) -> None:
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail=f"unknown dataset: {dataset_id}")


def _to_filters(p: RowsParams) -> dict[str, Any]:
    return {
        "q": p.q,
        "task_type": p.task_type,
        "register": p.register_,
        "length_band": p.length_band,
        "querent_context": p.querent_context,
        "spread_id": p.spread_id,
        "card_id": p.card_id,
        "orientation": p.orientation,
        "tier": p.tier,
        "ifd_min": p.ifd_min,
        "ifd_max": p.ifd_max,
    }


# --------------------------------------------------------------------------- #
# Public API (functions accept root for testability)
# --------------------------------------------------------------------------- #

def get_rows(root: Path, dataset_id: str, p: RowsParams) -> RowsResponse:
    """Paginated rows for the dataset under the given filters."""
    _require_dataset(dataset_id)
    rows = list(iter_matching(root, dataset_id, **_to_filters(p)))
    start = (p.page - 1) * p.page_size
    return RowsResponse(
        total=len(rows),
        page=p.page,
        page_size=p.page_size,
        rows=rows[start : start + p.page_size],
    )


def get_row(root: Path, dataset_id: str, id_path: str) -> dict[str, Any]:
    """Single raw record by the dataset's primary key; 404 when absent."""
    _require_dataset(dataset_id)
    _rel, pk, _search = DATASETS[dataset_id]
    if pk == "card_orientation":
        parts = id_path.split("/")
        if len(parts) != 2:
            raise HTTPException(status_code=404, detail=f"card not found: {id_path}")
        try:
            card_id = int(parts[0])
        except ValueError:
            raise HTTPException(status_code=404, detail=f"card not found: {id_path}")
        for row, _tier in iter_rows(root, dataset_id):
            if row.get("card_id") == card_id and row.get("orientation") == parts[1]:
                return row
        raise HTTPException(status_code=404, detail=f"card not found: {id_path}")
    if pk == "card_id":
        try:
            card_id = int(id_path)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"card not found: {id_path}")
        for row, _tier in iter_rows(root, dataset_id):
            if row.get("card_id") == card_id:
                return row
        raise HTTPException(status_code=404, detail=f"card not found: {id_path}")
    for row, _tier in iter_rows(root, dataset_id):
        if row.get(pk) == id_path:
            return row
    raise HTTPException(status_code=404, detail=f"no {pk} {id_path!r} in {dataset_id}")


def iter_export_lines(root: Path, dataset_id: str, p: RowsParams) -> Iterator[str]:
    """Canonical JSONL lines for every matching row (pagination ignored)."""
    for row in iter_matching(root, dataset_id, **_to_filters(p)):
        yield dumps_canonical(row) + "\n"


# --------------------------------------------------------------------------- #
# Routes (mounted by server.py in A.6)
# --------------------------------------------------------------------------- #

router = APIRouter(tags=["filtering"])


@router.get("/api/rows/{dataset_id}", response_model=RowsResponse)
def api_rows(dataset_id: str, params: Annotated[RowsParams, Query()]) -> RowsResponse:
    return get_rows(REPO_ROOT, dataset_id, params)


@router.get("/api/rows/{dataset_id}/{id_path:path}")
def api_row(dataset_id: str, id_path: str) -> dict[str, Any]:
    return get_row(REPO_ROOT, dataset_id, id_path)


@router.get("/api/export/{dataset_id}")
def api_export(dataset_id: str, params: Annotated[RowsParams, Query()]) -> StreamingResponse:
    _require_dataset(dataset_id)
    return StreamingResponse(
        iter_export_lines(REPO_ROOT, dataset_id, params),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f"attachment; filename={dataset_id}.jsonl"},
    )


@router.head("/api/export/{dataset_id}")
def api_export_head(dataset_id: str, params: Annotated[RowsParams, Query()]) -> Response:
    """HEAD mirror of the export (same headers, no body) for curl -sI checks."""
    _require_dataset(dataset_id)
    return Response(
        content=b"",
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f"attachment; filename={dataset_id}.jsonl"},
    )
