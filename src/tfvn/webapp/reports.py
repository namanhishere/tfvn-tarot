"""Report contents API — plan gap closure for the B.7 Reports view.

A.2's catalog exposes report metadata only (rows / size / sha256); this module
serves the actual parsed JSON of the six pipeline reports:

    datasets/filter_report.json
    datasets/coverage_report.json
    datasets/split_stats.json      (datasets/splits.json nested as metadata)
    datasets/ablation_report.json
    kb/w2_2_gate_report.json
    kb/spreads_discrimination_report.json

The files are small (≤ ~0.5 MB) and are not modified by anything else, so each
request reads from disk directly — no cache needed. All access is defensive:
a missing file drops out of the listing and 404s in the content endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]

router = APIRouter(tags=["reports"])


class ReportInfo(BaseModel):
    id: str
    path: str
    title: str
    splits_path: Optional[str] = None


class ReportListResponse(BaseModel):
    reports: list[ReportInfo]


#: Registry in listing order. ``splits.json`` is nested as metadata on the
#: ``split_stats`` entry so the listing stays at exactly the six reports.
_REPORTS: list[dict[str, Any]] = [
    {
        "id": "filter_report",
        "path": "datasets/filter_report.json",
        "title": "Filter report — dedup cascade, IFD, L3/L4 judge layers",
    },
    {
        "id": "coverage_report",
        "path": "datasets/coverage_report.json",
        "title": "Coverage report — task types, safety pairs, dedup cascade",
    },
    {
        "id": "split_stats",
        "path": "datasets/split_stats.json",
        "title": "Split statistics — train/val/test sizes and acceptance",
        "splits_path": "datasets/splits.json",
    },
    {
        "id": "ablation_report",
        "path": "datasets/ablation_report.json",
        "title": "Ablation report — filter-layer impact analysis",
    },
    {
        "id": "w2_2_gate_report",
        "path": "kb/w2_2_gate_report.json",
        "title": "W2.2 gate report — reversed-synthesis quality gates",
    },
    {
        "id": "spreads_discrimination_report",
        "path": "kb/spreads_discrimination_report.json",
        "title": "Spreads discrimination report — per-spread top-1 rates",
    },
]


def _available_reports(root: Path) -> list[dict[str, Any]]:
    """Registry entries whose file exists on disk (present-if-exists)."""
    return [entry for entry in _REPORTS if (root / entry["path"]).exists()]


def get_report_list(root: Path = REPO_ROOT) -> ReportListResponse:
    return ReportListResponse(
        reports=[ReportInfo(**entry) for entry in _available_reports(root)]
    )


def get_report(root: Path, report_id: str) -> dict[str, Any]:
    """Parsed contents of one report; 404 for unknown id or missing file."""
    entry = next(
        (e for e in _available_reports(root) if e["id"] == report_id), None
    )
    if entry is None:
        raise HTTPException(404, f"unknown report: {report_id}")
    path = root / entry["path"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise HTTPException(404, f"report file missing: {entry['path']}")
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"report {report_id} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise HTTPException(500, f"report {report_id} is not a JSON object")
    return data


# --------------------------------------------------------------------------- #
# Routes (mounted by server.py in A.6)
# --------------------------------------------------------------------------- #


@router.get("/api/reports", response_model=ReportListResponse)
def api_reports(root: Path = REPO_ROOT) -> ReportListResponse:
    return get_report_list(root)


@router.get("/api/reports/{report_id}")
def api_report(report_id: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    return get_report(root, report_id)
