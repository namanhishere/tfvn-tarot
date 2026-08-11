"""Artifact catalog and hash-integrity checks (todo A.2).

Registry table over ``kb/`` and ``datasets/`` with per-artifact metadata
(rows, size_bytes, informational raw-bytes sha256, schema keys) plus
canonical-serialisation hash integrity that REPLICATES the build scripts:

- ``kb/cards.jsonl`` vs ``kb/CARDS_HASH.txt``: ``build_wave2_api.py:588``
  ``sha256(dumps_canonical(rows).encode("utf-8"))`` over parsed rows in
  file order.
- filtered_core + filtered_bulk vs ``datasets/DATASET_HASH.txt``:
  ``build_wave3.py:1377-1381`` — concat tiers, sort by ``example_id``, then
  ``sha256("\n".join(dumps_canonical(r) ...).encode("utf-8"))``.

Catalog and checks are cached by input (mtime,size) of the source files;
call :func:`invalidate` after a pipeline re-run. All access is defensive —
missing gitignored raw files appear as present-if-exists (rows 0, size 0,
sha256 None) and never raise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..serialise import dumps_canonical, read_jsonl
from .catalog_registry import REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# Pydantic response models
# --------------------------------------------------------------------------- #

class ArtifactInfo(BaseModel):
    id: str
    path: str
    kind: str
    tag: str
    rows: Optional[int] = None
    size_bytes: int = 0
    sha256: Optional[str] = None
    schema_keys: Optional[list[str]] = None


class CatalogResponse(BaseModel):
    artifacts: list[ArtifactInfo]


class HashCheck(BaseModel):
    id: str
    method: str = "canonical"
    matches: bool
    computed_digest: Optional[str] = None
    recorded_digest: Optional[str] = None


class HashCheckResponse(BaseModel):
    cards_match: bool
    dataset_match: bool
    checks: list[HashCheck]

# --------------------------------------------------------------------------- #
# Per-artifact inspection
# --------------------------------------------------------------------------- #

def _jsonl_rows_and_keys(data: bytes) -> tuple[int, Optional[list[str]]]:
    """Row count + first-row keys from raw JSONL bytes (one obj per line)."""
    keys: Optional[list[str]] = None
    count = 0
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        count += 1
        if keys is None:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys = list(row.keys()) if isinstance(row, dict) else None
    return count, keys


def _json_top_keys(data: bytes) -> Optional[list[str]]:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(obj, dict):
        return list(obj.keys())
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return list(obj[0].keys())
    return None


def _inspect(entry: dict[str, str], root: Path) -> ArtifactInfo:
    rel = entry["path"]
    missing = ArtifactInfo(
        id=rel, path=rel, kind=entry["kind"], tag=entry["tag"],
        rows=0, size_bytes=0, sha256=None, schema_keys=None,
    )
    try:
        data = (root / rel).read_bytes()
    except OSError:
        return missing
    size = len(data)
    info = ArtifactInfo(
        id=rel, path=rel, kind=entry["kind"], tag=entry["tag"],
        size_bytes=size, sha256=hashlib.sha256(data).hexdigest(),
    )
    if entry["kind"] == "jsonl":
        info.rows, info.schema_keys = _jsonl_rows_and_keys(data)
    elif entry["kind"] == "json":
        info.schema_keys = _json_top_keys(data)
    return info

# --------------------------------------------------------------------------- #
# Canonical hash checks (replicate build scripts EXACTLY)
# --------------------------------------------------------------------------- #

def _cards_digest(root: Path) -> Optional[str]:
    path = root / "kb" / "cards.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    return hashlib.sha256(dumps_canonical(rows).encode("utf-8")).hexdigest()


def _datasets_digest(root: Path) -> Optional[str]:
    core_path = root / "datasets" / "filtered_core.jsonl"
    bulk_path = root / "datasets" / "filtered_bulk.jsonl"
    core = read_jsonl(core_path) if core_path.exists() else []
    bulk = read_jsonl(bulk_path) if bulk_path.exists() else []
    combined = core + bulk
    canonical = "\n".join(
        dumps_canonical(r) for r in sorted(combined, key=lambda r: r["example_id"])
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _recorded_digest(root: Path, rel: str) -> Optional[str]:
    try:
        text = (root / rel).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text.splitlines()[0].strip() if text else None


def _compute_checks(root: Path) -> list[HashCheck]:
    targets = [
        ("cards", _cards_digest(root), _recorded_digest(root, "kb/CARDS_HASH.txt")),
        ("datasets", _datasets_digest(root), _recorded_digest(root, "datasets/DATASET_HASH.txt")),
    ]
    return [
        HashCheck(
            id=tid,
            method="canonical",
            matches=bool(rec and comp == rec),
            computed_digest=comp,
            recorded_digest=rec,
        )
        for tid, comp, rec in targets
    ]

# --------------------------------------------------------------------------- #
# Input-keyed cache + public API
# --------------------------------------------------------------------------- #

_CACHE: dict[str, Any] = {"key": None, "artifacts": None, "checks": None}


def _input_key(root: Path) -> tuple:
    key: list[tuple] = []
    for entry in REGISTRY:
        try:
            st = (root / entry["path"]).stat()
            key.append((entry["path"], st.st_mtime_ns, st.st_size))
        except OSError:
            key.append((entry["path"], None))
    return tuple(key)


def _state(root: Path) -> tuple[list[ArtifactInfo], list[HashCheck]]:
    key = _input_key(root)
    if _CACHE["key"] != key or _CACHE["artifacts"] is None:
        _CACHE["key"] = key
        _CACHE["artifacts"] = [_inspect(e, root) for e in REGISTRY]
        _CACHE["checks"] = _compute_checks(root)
    return _CACHE["artifacts"], _CACHE["checks"]


def invalidate() -> None:
    """Drop the cache; next call recomputes from disk (post-run refresh)."""
    _CACHE["key"] = None
    _CACHE["artifacts"] = None
    _CACHE["checks"] = None


def get_catalog(root: Path = REPO_ROOT) -> CatalogResponse:
    artifacts, _ = _state(root)
    return CatalogResponse(artifacts=artifacts)


def get_hashcheck(root: Path = REPO_ROOT) -> HashCheckResponse:
    artifacts, checks = _state(root)
    return HashCheckResponse(
        cards_match=checks[0].matches,
        dataset_match=checks[1].matches,
        checks=checks,
    )

# --------------------------------------------------------------------------- #
# Routes (mounted by server.py in A.6)
# --------------------------------------------------------------------------- #

router = APIRouter(tags=["catalog"])


@router.get("/api/catalog", response_model=CatalogResponse)
def api_catalog() -> CatalogResponse:
    return get_catalog()


@router.get("/api/hashcheck", response_model=HashCheckResponse)
def api_hashcheck() -> HashCheckResponse:
    return get_hashcheck()
