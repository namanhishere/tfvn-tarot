"""A.5 whitelisted pipeline script runner — router, single-flight, history.

Endpoints:
  GET  /api/runs                    whitelist + preflight status + running state
  POST /api/runs/{script_id}        start a run (tiered confirm, validated opts)
  GET  /api/runs/{run_id}/log       live log tail (?offset=N) from the ring
  POST /api/runs/{run_id}/kill      terminate the subprocess
  POST /api/runs/{run_id}/ack       acknowledge a boot-time orphaned run
  GET  /api/runs/history            file-backed run history (newest first)

Runtime discipline: single-flight (second start -> 409), boot-time orphan
reconciliation (persisted ``running`` rows with no live subprocess are marked
``orphaned`` and block new runs until acknowledged), per-tier timeout, kill
with 5 s grace. History persists to ``logs/webapp_runs.jsonl`` (gitignored);
log files live under ``logs/webapp_runs/``. Requires uvicorn workers=1 without
``--reload`` (see plan) so subprocesses are never orphaned by a worker restart.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..serialise import dumps_canonical
from .runs_gates import (
    display_argv,
    effective_argv,
    gate_derivation,
    preflight_checks,
    preflight_refusal,
    validate_options,
)
from .runs_runner import RingLog, build_scrubber, kill_process, run_process
from .runs_whitelist import REPO_ROOT, WHITELIST, RunSpec, resolve_script_id

router = APIRouter()

HISTORY_RELPATH = "logs/webapp_runs.jsonl"
LOG_DIR_RELPATH = "logs/webapp_runs"


class RunStartRequest(BaseModel):
    confirm: bool = False
    accept_cost: bool = False
    options: dict[str, Any] = {}
    fresh_run: bool = False


# --------------------------------------------------------------------------- #
# In-memory state (uvicorn workers=1, no --reload — see module docstring)
# --------------------------------------------------------------------------- #


@dataclass
class RunHandle:
    run_id: str
    spec: RunSpec
    argv: list[str]
    options: dict[str, Any]
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    proc: Optional[asyncio.subprocess.Process] = None
    task: Optional[asyncio.Task] = None
    exit_code: Optional[int] = None
    timed_out: bool = False
    killed: bool = False
    status: str = "running"
    gates_passed: Optional[bool] = None
    gate_detail: Optional[str] = None
    ring: RingLog = field(default_factory=RingLog)
    log_path: Optional[Path] = None
    log_fh: Optional[Any] = None
    scrub: Optional[Any] = None
    confirm: bool = False
    accept_cost: bool = False
    fresh_run: bool = False
    moved_to: Optional[str] = None
    git_head: str = ""
    git_dirty: bool = False

    def emit(self, line: str) -> None:
        """Exception-safe line sink: ring buffer + run log file (scrubbed)."""
        try:
            scrubbed = self.scrub(line) if self.scrub else line
        except Exception:
            scrubbed = line
        self.ring.append(scrubbed)
        if self.log_fh is not None:
            try:
                self.log_fh.write(scrubbed + "\n")
                self.log_fh.flush()
            except OSError:
                pass

    def close_log(self) -> None:
        if self.log_fh is not None:
            try:
                self.log_fh.close()
            except OSError:
                pass
            self.log_fh = None


_HANDLES: dict[str, RunHandle] = {}
_CURRENT: Optional[RunHandle] = None
_HISTORY: list[dict[str, Any]] = []
_RECONCILED = False
_ORPHANED: list[str] = []  # run_ids awaiting acknowledgement


# --------------------------------------------------------------------------- #
# History persistence (file-backed, canonical JSONL)
# --------------------------------------------------------------------------- #


def _history_path(root: Path = REPO_ROOT) -> Path:
    return root / HISTORY_RELPATH


def _load_history(root: Path = REPO_ROOT) -> None:
    global _HISTORY
    p = _history_path(root)
    _HISTORY = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                _HISTORY.append(json.loads(line))
            except ValueError:
                continue


def _append_history(rec: dict[str, Any], root: Path = REPO_ROOT) -> None:
    _HISTORY.append(rec)
    p = _history_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(dumps_canonical(rec) + "\n")


def _rewrite_history(root: Path = REPO_ROOT) -> None:
    p = _history_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in _HISTORY:
            fh.write(dumps_canonical(rec) + "\n")


def _ensure_reconciled(root: Path = REPO_ROOT) -> None:
    """Boot-time orphan reconciliation, run once.

    Any persisted ``running`` row has no live subprocess in this process —
    after a restart the in-memory single-flight lock is gone, so an orphaned
    w23 could otherwise rewrite tracked ``kb/cards.jsonl`` while we start
    something new. Mark them ``orphaned`` and block new runs until the user
    acknowledges.
    """
    global _RECONCILED, _ORPHANED
    if _RECONCILED:
        return
    _RECONCILED = True
    changed = False
    for rec in _HISTORY:
        if rec.get("status") == "running":
            rec["status"] = "orphaned"
            rec["acknowledged"] = False
            _ORPHANED.append(rec["run_id"])
            changed = True
    if changed:
        _rewrite_history(root)


# --------------------------------------------------------------------------- #
# Git context + cache invalidation
# --------------------------------------------------------------------------- #


def _git_info(root: Path = REPO_ROOT) -> tuple[str, bool]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return head.stdout.strip(), bool(status.stdout.strip())
    except Exception:
        return "unknown", False


def _invalidate_caches() -> None:
    """Stats and catalog are input-cached; a completed run rewrote their inputs."""
    for modname in ("stats", "catalog"):
        try:
            mod = __import__(f"{__package__}.{modname}", fromlist=["invalidate"])
            mod.invalidate()
        except Exception:
            pass


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _record_from(handle: RunHandle) -> dict[str, Any]:
    return {
        "run_id": handle.run_id,
        "script_id": handle.spec.script_id,
        "label": handle.spec.label,
        "tier": handle.spec.tier,
        "argv": handle.argv,
        "options": handle.options,
        "confirm": handle.confirm,
        "accept_cost": handle.accept_cost,
        "fresh_run": handle.fresh_run,
        "moved_to": handle.moved_to,
        "status": handle.status,
        "exit_code": handle.exit_code,
        "gates_passed": handle.gates_passed,
        "gate_detail": handle.gate_detail,
        "killed": handle.killed,
        "timed_out": handle.timed_out,
        "started_at": _iso(handle.started_at),
        "finished_at": _iso(handle.finished_at) if handle.finished_at else None,
        "duration_s": (
            round(handle.finished_at - handle.started_at, 2)
            if handle.finished_at
            else None
        ),
        "git_head": handle.git_head,
        "git_dirty": handle.git_dirty,
        "modifies": list(handle.spec.modifies),
        "acknowledged": False,
    }


# --------------------------------------------------------------------------- #
# Runner orchestration
# --------------------------------------------------------------------------- #


async def _run_and_finalize(handle: RunHandle) -> None:
    global _CURRENT

    def _on_start(proc: asyncio.subprocess.Process) -> None:
        handle.proc = proc

    try:
        exit_code, timed_out = await run_process(
            handle.argv,
            cwd=str(REPO_ROOT),
            env=os.environ,
            emit=handle.emit,
            timeout_s=float(handle.spec.timeout_s),
            on_start=_on_start,
        )
        handle.exit_code = exit_code
        handle.timed_out = timed_out
        if handle.killed:
            handle.status = "killed"
        elif timed_out:
            handle.status = "timed_out"
        else:
            handle.status = "completed" if exit_code == 0 else "failed"
        if handle.exit_code is not None:
            handle.gates_passed, handle.gate_detail = gate_derivation(
                handle.spec, handle.exit_code
            )
        else:
            handle.gates_passed, handle.gate_detail = False, "process did not exit"
    except Exception as exc:  # runner crash must never wedge single-flight
        handle.status = "failed"
        handle.exit_code = -1
        handle.gates_passed, handle.gate_detail = False, f"runner error: {exc}"
    finally:
        handle.finished_at = time.time()
        handle.close_log()
        _append_history(_record_from(handle))
        _invalidate_caches()
        if _CURRENT is handle:
            _CURRENT = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get("/api/runs")
async def list_runs() -> dict[str, Any]:
    _ensure_reconciled()
    scripts = []
    for sid in sorted(WHITELIST):
        spec = WHITELIST[sid]
        scripts.append(
            {
                "script_id": sid,
                "label": spec.label,
                "description": spec.description,
                "tier": spec.tier,
                "argv": display_argv(spec),
                "options": {
                    key: {
                        "type": opt.opt_type,
                        "min": opt.minimum,
                        "max": opt.maximum,
                        "default": opt.default,
                        "required": opt.required,
                    }
                    for key, opt in spec.options.items()
                },
                "timeout_s": spec.timeout_s,
                "confirm_required": spec.confirm_required,
                "modifies": list(spec.modifies),
                "preflight": [
                    {"label": r.label, "ok": r.ok, "reason": r.reason}
                    for r in preflight_checks(spec)
                ],
            }
        )
    return {
        "scripts": scripts,
        "running": (
            {"run_id": _CURRENT.run_id, "script_id": _CURRENT.spec.script_id}
            if _CURRENT
            else None
        ),
        "orphaned_pending": list(_ORPHANED),
    }


@router.post("/api/runs/{script_id}")
async def start_run(script_id: str, req: RunStartRequest) -> dict[str, Any]:
    global _CURRENT
    _ensure_reconciled()
    script_id = resolve_script_id(script_id) or script_id
    if script_id not in WHITELIST:
        raise HTTPException(404, f"unknown script {script_id!r}")
    spec = WHITELIST[script_id]
    if _CURRENT is not None:
        raise HTTPException(
            409,
            f"a run is in progress ({_CURRENT.spec.script_id} / {_CURRENT.run_id})",
        )
    if _ORPHANED:
        raise HTTPException(
            409,
            "an orphaned run is pending acknowledgement"
            f" (POST /api/runs/{_ORPHANED[0]}/ack): {', '.join(_ORPHANED)}",
        )
    try:
        options = validate_options(spec, req.options)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    if spec.tier == "billed":
        if not (req.confirm and req.accept_cost):
            raise HTTPException(
                409,
                f"{script_id} is billed: confirmation and accept_cost=true are "
                "required",
            )
    elif spec.confirm_required and not req.confirm:
        raise HTTPException(
            409, f"confirmation (confirm=true) is required for {script_id}"
        )
    if req.fresh_run and script_id != "w32":
        raise HTTPException(422, "fresh_run is only supported for w32")
    refusal = preflight_refusal(spec)
    if refusal:
        raise HTTPException(409, refusal)

    moved: Optional[str] = None
    if req.fresh_run:
        gen = REPO_ROOT / "datasets" / "raw" / "generated.jsonl"
        if gen.exists():
            backup = gen.with_name(
                f"generated.jsonl.{time.strftime('%Y%m%dT%H%M%S')}.bak"
            )
            gen.rename(backup)
            moved = str(backup.relative_to(REPO_ROOT))

    argv = effective_argv(spec, options)
    run_id = f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    git_head, git_dirty = _git_info()
    handle = RunHandle(
        run_id=run_id,
        spec=spec,
        argv=argv,
        options=options,
        confirm=req.confirm,
        accept_cost=req.accept_cost,
        fresh_run=req.fresh_run,
        moved_to=moved,
        git_head=git_head,
        git_dirty=git_dirty,
    )
    handle.scrub = build_scrubber()
    log_dir = REPO_ROOT / LOG_DIR_RELPATH
    log_dir.mkdir(parents=True, exist_ok=True)
    handle.log_path = log_dir / f"{run_id}.log"
    handle.log_fh = handle.log_path.open("a", encoding="utf-8", buffering=1)
    _HANDLES[run_id] = handle
    _CURRENT = handle
    handle.task = asyncio.create_task(_run_and_finalize(handle))
    return {
        "run_id": run_id,
        "script_id": script_id,
        "tier": spec.tier,
        "status": "running",
        "argv": argv,
    }


@router.get("/api/runs/{run_id}/log")
async def run_log(run_id: str, offset: int = 0) -> dict[str, Any]:
    handle = _HANDLES.get(run_id)
    if handle is None:
        raise HTTPException(404, f"unknown run {run_id!r}")
    lines, next_offset, truncated = handle.ring.read(offset)
    return {
        "run_id": run_id,
        "status": handle.status,
        "offset": next_offset,
        "truncated": truncated,
        "lines": lines,
    }


@router.post("/api/runs/{run_id}/kill")
async def kill_run(run_id: str) -> dict[str, Any]:
    handle = _HANDLES.get(run_id)
    if handle is None:
        raise HTTPException(404, f"unknown run {run_id!r}")
    if handle.status != "running":
        raise HTTPException(409, f"run {run_id} is not running (status={handle.status})")
    handle.killed = True
    proc = handle.proc
    if proc is None:
        # Runner has not spawned yet — wait briefly for on_start to set it.
        for _ in range(50):
            if handle.proc is not None:
                break
            await asyncio.sleep(0.02)
        proc = handle.proc
    if proc is not None:
        await kill_process(proc)
    return {"run_id": run_id, "killed": True}


@router.post("/api/runs/{run_id}/ack")
async def ack_run(run_id: str) -> dict[str, Any]:
    _ensure_reconciled()
    rec = next((r for r in _HISTORY if r.get("run_id") == run_id), None)
    if rec is None:
        raise HTTPException(404, f"unknown run {run_id!r}")
    if rec.get("status") != "orphaned" or rec.get("acknowledged"):
        raise HTTPException(409, f"run {run_id} is not a pending orphaned run")
    rec["acknowledged"] = True
    if run_id in _ORPHANED:
        _ORPHANED.remove(run_id)
    _rewrite_history()
    return {
        "run_id": run_id,
        "acknowledged": True,
        "orphaned_pending": list(_ORPHANED),
    }


@router.get("/api/runs/history")
async def run_history() -> dict[str, Any]:
    _ensure_reconciled()
    return {"runs": list(reversed(_HISTORY))}


_load_history()
