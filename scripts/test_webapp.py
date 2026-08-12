#!/usr/bin/env python3
"""End-to-end smoke test for the dataset-viewer webapp (plan todo C.2).

FastAPI ``TestClient`` against the REAL app (``tfvn.webapp.server.app``) with
REAL repo data for every read-only endpoint, plus the run-lifecycle cases —
billed-409 and single-flight-409 — and a real safe ``kb_rebuild``
(``build_wave1``) run that must complete exit 0 and write a history row with
``git_head``.

The kb_rebuild e2e calls the async route handler DIRECTLY under
``asyncio.run()`` with the real runner — never a bare TestClient POST: a
TestClient closes its per-request portal after each request, cancelling the
run's background task mid-lifecycle and orphaning the finalizer (history
written as ``running`` / ``exit None``). Driving the handler under a loop we
keep alive until ``runs._CURRENT`` clears lets the subprocess finish, the
history row get written, and the stats/catalog caches invalidate — exactly as
under uvicorn.

Mirrors ``scripts/test_wave2_api.py``: sys.path inserts for ``src/`` and
``scripts/``, a PASS/FAIL print per assertion, exit 0 only if everything
passes. The only mutation is the safe deterministic ``build_wave1`` re-run
(byte-identical rewrite of ``kb/``); ``logs/webapp_runs.jsonl`` and the
per-run logs are snapshotted before and restored after so no residue is left.

Usage:
  .venv/bin/python scripts/test_webapp.py

Exit code 0 = all checks green.
"""

from __future__ import annotations

import asyncio  # noqa: ANYIO_OK — drive the app's asyncio route handlers directly
import json
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tfvn.webapp import runs  # noqa: E402
from tfvn.webapp.runs_gates import gate_derivation  # noqa: E402
from tfvn.webapp.runs_whitelist import WHITELIST  # noqa: E402
from tfvn.webapp.server import app  # noqa: E402

LOG_HISTORY = ROOT / "logs" / "webapp_runs.jsonl"
LOG_DIR = ROOT / "logs" / "webapp_runs"

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    """Print PASS/FAIL; append failures for the final verdict."""
    if cond:
        print(f"  PASS {label}")
    else:
        print(f"  FAIL {label} — {detail}")
        FAILS.append(f"{label}: {detail}")


def _j(value: dict | list | str | int | float | bool | None) -> str:
    """Compact stable JSON for FAIL details."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- #
# Read-only API sections (TestClient)
# --------------------------------------------------------------------------- #


def _api_health(client: TestClient) -> None:
    print("[1/13] /api/health")
    r = client.get("/api/health")
    check("health status 200", r.status_code == 200, f"status={r.status_code}")
    check("health body ok", r.json() == {"status": "ok"}, _j(r.json()))


def _api_catalog(client: TestClient) -> None:
    print("[2/13] /api/catalog counts")
    artifacts = client.get("/api/catalog").json()["artifacts"]
    by_path = {a["path"]: a for a in artifacts}
    for rel, expected in (
        ("kb/cards.jsonl", 156),
        ("datasets/filtered_core.jsonl", 5000),
        ("datasets/filtered_bulk.jsonl", 8571),
        ("datasets/anchor/anchor_readings.jsonl", 30),
        ("kb/spreads.jsonl", 21),
    ):
        got = by_path.get(rel)
        check(
            f"catalog {rel} rows == {expected}",
            got is not None and got["rows"] == expected,
            f"got {got}",
        )


def _api_hashcheck(client: TestClient) -> None:
    print("[3/13] /api/hashcheck (canonical)")
    hc = client.get("/api/hashcheck").json()
    check("hashcheck cards_match", hc["cards_match"] is True, _j(hc))
    check("hashcheck dataset_match", hc["dataset_match"] is True, _j(hc))
    check("hashcheck method canonical", hc["checks"][0]["method"] == "canonical", _j(hc))


def _api_stats(client: TestClient) -> None:
    print("[4/13] /api/stats totals")
    st = client.get("/api/stats").json()
    src = st.get("source") or {}
    check("stats source.total == 13571", src.get("total") == 13571, _j(src))
    tc = st.get("tier_counts") or {}
    check("stats tier_counts core == 5000", tc.get("core") == 5000, _j(tc))
    check("stats tier_counts bulk == 8571", tc.get("bulk") == 8571, _j(tc))
    for key in (
        "source", "tier_counts", "distributions", "per_card", "ifd",
        "splits", "kb", "spreads", "anchor", "total_reversed_percent",
    ):
        check(f"stats key {key!r} present", key in st, f"keys={sorted(st)}")


def _api_rows_safety(client: TestClient) -> None:
    print("[5/13] /api/rows/all_sft?task_type=safety&page_size=200")
    payload = client.get(
        "/api/rows/all_sft", params={"task_type": "safety", "page_size": 200}
    ).json()
    total = payload["total"]
    rows = payload["rows"]
    missing = sum(1 for row in rows if not row.get("matched_pair_id"))
    check("safety rows total > 0", total > 0, f"total={total}")
    check("safety rows total == 107 (real-data drift signal)", total == 107, f"total={total}")
    check(
        "all returned safety rows carry matched_pair_id",
        missing == 0,
        f"{missing}/{len(rows)} missing",
    )


def _api_rows_tier_core(client: TestClient) -> None:
    print("[6/13] /api/rows/all_sft?tier=core")
    payload = client.get("/api/rows/all_sft", params={"tier": "core", "page_size": 1}).json()
    check("tier=core total == 5000", payload["total"] == 5000, f"total={payload['total']}")


def _api_export(client: TestClient) -> None:
    print("[7/13] /api/export/filtered_core?task_type=reading")
    rows_total = client.get(
        "/api/rows/filtered_core", params={"task_type": "reading", "page_size": 1}
    ).json()["total"]
    exp = client.get("/api/export/filtered_core", params={"task_type": "reading"})
    check("export status 200", exp.status_code == 200, f"status={exp.status_code}")
    cdisp = exp.headers.get("content-disposition", "")
    check("export content-disposition attachment", "attachment" in cdisp, cdisp)
    lines = [ln for ln in exp.text.splitlines() if ln.strip()]
    bad: list[str] = []
    for ln in lines:
        try:
            json.loads(ln)
        except json.JSONDecodeError:
            bad.append(ln[:80])
    check("export every line parses via json.loads", not bad, f"{len(bad)} bad: {bad[:3]}")
    check(
        "export line count == filtered rows total",
        len(lines) == rows_total,
        f"lines={len(lines)} rows_total={rows_total}",
    )


def _api_runs_listing(client: TestClient) -> None:
    print("[8/13] /api/runs whitelist")
    scripts = client.get("/api/runs").json()["scripts"]
    check("whitelist lists 11 scripts", len(scripts) == 11, f"got {len(scripts)}")
    tiers = Counter(s["tier"] for s in scripts)
    check(
        "tier counts safe==5 slow==3 billed==3",
        dict(tiers) == {"safe": 5, "slow": 3, "billed": 3},
        _j(dict(tiers)),
    )
    w21 = next((s for s in scripts if s["script_id"] == "w21"), None)
    check("w21 entry present", w21 is not None, _j([s["script_id"] for s in scripts]))
    if w21 is not None:
        check("w21 effective argv carries --dry-run", "--dry-run" in w21["argv"], _j(w21["argv"]))


def _api_billed_409(client: TestClient) -> None:
    print("[9/13] billed run without accept_cost -> 409")
    r = client.post("/api/runs/w22", json={"confirm": True})
    check("w22 confirm-only POST -> 409", r.status_code == 409, f"status={r.status_code} body={r.text}")


# --------------------------------------------------------------------------- #
# Run lifecycle (direct-async — see module docstring)
# --------------------------------------------------------------------------- #


class _FakeProc:
    """Minimal stand-in for ``asyncio.subprocess.Process`` (kill/signal paths)."""

    def __init__(self) -> None:
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


class FakeRunner:
    """Injected ``run_process``: records argv, streams lines, holds the
    subprocess "alive" until :attr:`release` is set (thread-safe)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.release = threading.Event()

    async def run(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        emit: Callable[[str], None],
        timeout_s: float | None = None,
        kill_grace: float = 5.0,
        on_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> tuple[int, bool]:
        self.calls.append(list(argv))
        if on_start is not None:
            on_start(_FakeProc())  # type: ignore[arg-type]
        emit("fake line one")
        emit("fake line two")
        while not self.release.is_set():
            await asyncio.sleep(0.005)
        return 0, False


def _reset_runs_state() -> None:
    runs._HISTORY = []
    runs._RECONCILED = False
    runs._ORPHANED = []
    runs._CURRENT = None
    runs._HANDLES = {}


async def _wait_finalized(timeout: float = 120.0) -> bool:
    """Keep the loop alive until the active run's finalizer completes."""
    deadline = time.time() + timeout
    while runs._CURRENT is not None and time.time() < deadline:
        await asyncio.sleep(0.02)
    return runs._CURRENT is None


def _api_concurrent_409() -> None:
    print("[10/13] second concurrent run start -> 409")
    _reset_runs_state()
    runner = FakeRunner()
    real_run_process = runs.run_process
    runs.run_process = runner.run
    try:
        async def scenario() -> None:
            first = await runs.start_run("w21", runs.RunStartRequest())
            check("first start returns running", first["status"] == "running", _j(first))
            try:
                await runs.start_run("w21", runs.RunStartRequest())
                check("second start while running -> 409", False, "second start did not raise")
            except HTTPException as exc:
                check(
                    "second start while running -> 409",
                    exc.status_code == 409,
                    f"status={exc.status_code} detail={exc.detail}",
                )
            finally:
                runner.release.set()
                await _wait_finalized(timeout=10.0)

        asyncio.run(scenario())
    finally:
        runs.run_process = real_run_process


def _api_kb_rebuild_e2e(client: TestClient) -> None:
    print("[11/13] safe kb_rebuild e2e (direct-async, real runner)")
    _reset_runs_state()

    async def scenario() -> None:
        started = await runs.start_run("kb_rebuild", runs.RunStartRequest(confirm=True))
        run_id = started["run_id"]
        check(
            "kb_rebuild start -> running build_wave1",
            started["status"] == "running" and started["script_id"] == "build_wave1",
            _j(started),
        )
        finalized = await _wait_finalized(timeout=300.0)
        check(
            "kb_rebuild finalizer completed (not orphaned)",
            finalized,
            "runs._CURRENT still set after 300 s",
        )
        history = await runs.run_history()
        rec = next((x for x in history["runs"] if x.get("run_id") == run_id), None)
        check("kb_rebuild history row written", rec is not None, _j(history))
        if rec is not None:
            check("kb_rebuild exit_code == 0", rec.get("exit_code") == 0, _j(rec))
            check("kb_rebuild status == completed", rec.get("status") == "completed", _j(rec))
            check("kb_rebuild gates_passed == True", rec.get("gates_passed") is True, _j(rec))
            gh = rec.get("git_head")
            check(
                "kb_rebuild history carries git_head",
                isinstance(gh, str) and len(gh) == 40,
                _j(rec),
            )

    asyncio.run(scenario())
    hc = client.get("/api/hashcheck").json()
    check(
        "kb_rebuild hashcheck still green after",
        hc["cards_match"] is True and hc["dataset_match"] is True,
        _j(hc),
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "kb/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    check(
        "kb_rebuild left kb/ git-clean (byte-identical rewrite)",
        dirty.stdout.strip() == "",
        f"dirty: {dirty.stdout.strip()[:200]}",
    )


def _api_gate_fixtures() -> None:
    print("[12/13] gate_derivation unit fixtures (no real w34/w22 runs)")
    ok, detail = gate_derivation(WHITELIST["w34-full"], 0)
    check("w34-full real report -> gates_passed", ok is True, _j([ok, detail]))
    ok, detail = gate_derivation(WHITELIST["w22"], 0)
    check(
        "w22 real report -> gates_passed from aggregate",
        ok is True and "failed_gate=0" in detail,
        _j([ok, detail]),
    )
    ok, detail = gate_derivation(WHITELIST["build_wave1"], 0)
    check("build_wave1 exit-0 -> gates_passed", ok is True and detail == "exit code 0", _j([ok, detail]))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "datasets").mkdir(parents=True)
        report = root / "datasets" / "filter_report.json"
        report.write_text(
            json.dumps(
                {"acceptance": {"l1_dedup": True, "l2_ifd": True, "l3_deita": True, "l4_judge": True}}
            ),
            encoding="utf-8",
        )
        ok, detail = gate_derivation(WHITELIST["w34-full"], 0, root=root)
        check("w34 fixture all-pass acceptance -> gates_passed", ok is True and "4/4" in detail, _j([ok, detail]))
        report.write_text(
            json.dumps({"acceptance": {"l1_dedup": True, "l2_ifd": False}}),
            encoding="utf-8",
        )
        ok, detail = gate_derivation(WHITELIST["w34-full"], 0, root=root)
        check(
            "w34 fixture failing acceptance -> gates_passed false despite exit 0",
            ok is False and "l2_ifd" in detail,
            _j([ok, detail]),
        )

        (root / "kb").mkdir(parents=True)
        w22_report = root / "kb" / "w2_2_gate_report.json"
        w22_report.write_text(
            json.dumps({"aggregate": {"failed_gate": 0}, "negative_control_rejection_rate": 0.95}),
            encoding="utf-8",
        )
        ok, detail = gate_derivation(WHITELIST["w22"], 0, root=root)
        check("w22 fixture failed_gate==0 -> gates_passed", ok is True, _j([ok, detail]))
        w22_report.write_text(
            json.dumps({"aggregate": {"failed_gate": 1}, "negative_control_rejection_rate": 0.95}),
            encoding="utf-8",
        )
        ok, detail = gate_derivation(WHITELIST["w22"], 0, root=root)
        check("w22 fixture failed_gate!=0 -> gates_passed false", ok is False, _j([ok, detail]))
        w22_report.write_text(
            json.dumps({"aggregate": {"failed_gate": 0}, "negative_control_rejection_rate": 0.5}),
            encoding="utf-8",
        )
        ok, detail = gate_derivation(WHITELIST["w22"], 0, root=root)
        check("w22 fixture rejection<0.8 -> gates_passed false", ok is False, _j([ok, detail]))


def _api_reports(client: TestClient) -> None:
    print("[13/13] /api/reports")
    reports = client.get("/api/reports").json()["reports"]
    check("reports lists 6 entries", len(reports) == 6, f"got {len(reports)}")
    ids = [r["id"] for r in reports]
    expected = [
        "filter_report", "coverage_report", "split_stats", "ablation_report",
        "w2_2_gate_report", "spreads_discrimination_report",
    ]
    check("reports ids in registry order", ids == expected, _j(ids))


# --------------------------------------------------------------------------- #
# Runtime-state hygiene (gitignored logs must be restored after the e2e)
# --------------------------------------------------------------------------- #


def _snapshot_runtime() -> tuple[bytes | None, dict[str, bytes]]:
    hist = LOG_HISTORY.read_bytes() if LOG_HISTORY.exists() else None
    logs: dict[str, bytes] = {}
    if LOG_DIR.is_dir():
        for p in LOG_DIR.glob("*.log"):
            logs[p.name] = p.read_bytes()
    return hist, logs


def _restore_runtime(snapshot: tuple[bytes | None, dict[str, bytes]]) -> None:
    hist, logs = snapshot
    if hist is None:
        LOG_HISTORY.unlink(missing_ok=True)
    else:
        LOG_HISTORY.write_bytes(hist)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for p in LOG_DIR.glob("*.log"):
        if p.name not in logs:
            p.unlink()
    for name, data in logs.items():
        (LOG_DIR / name).write_bytes(data)


def main() -> int:
    snapshot = _snapshot_runtime()
    try:
        with TestClient(app) as client:
            _api_health(client)
            _api_catalog(client)
            _api_hashcheck(client)
            _api_stats(client)
            _api_rows_safety(client)
            _api_rows_tier_core(client)
            _api_export(client)
            _api_runs_listing(client)
            _api_billed_409(client)
            _api_concurrent_409()
            _api_kb_rebuild_e2e(client)
            _api_gate_fixtures()
            _api_reports(client)
    finally:
        _restore_runtime(snapshot)
        _reset_runs_state()

    print()
    if FAILS:
        print("RESULT: FAIL")
        for f in FAILS:
            print("  -", f)
        return 1
    print("RESULT: PASS — all smoke checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
