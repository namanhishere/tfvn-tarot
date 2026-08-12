"""Runs module tests (todo A.5 / C.1).

Verifies argv exactness for every whitelist entry (incl. the safe
``--only w21 --dry-run`` vs billed ``--only w22`` split), tier gating,
single-flight 409, boot-time orphan reconciliation, pre-flight refusal naming
the exact missing path, the fake-command runner (exit code / log / kill /
history / cache invalidation) and per-script ``gate_derivation`` from the A.5
table — including the w22 rule derived from ``aggregate.failed_gate`` +
``negative_control_rejection_rate``.

Router-level tests inject a :class:`FakeRunner` in place of ``run_process``
so no real script, no billing and no subprocess of ``build_wave*.py`` ever
happens. History and per-run logs land under the ``tmp_path`` fixture root.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import Counter

import pytest

from tfvn.webapp import runs, runs_gates, runs_whitelist
from tfvn.webapp.runs_whitelist import WHITELIST, resolve_script_id
from tfvn.webapp.runs_gates import (
    display_argv,
    effective_argv,
    gate_derivation,
    preflight_checks,
    preflight_refusal,
    validate_options,
)

# --------------------------------------------------------------------------- #
# argv / whitelist table (exact, from the A.5 table in the plan)
# --------------------------------------------------------------------------- #

EXPECTED_ARGV = {
    "build_wave1": ["scripts/build_wave1.py"],
    "w21": ["scripts/build_wave2_api.py", "--only", "w21", "--dry-run"],
    "w23": ["scripts/build_wave2_api.py", "--only", "w23", "--dry-run"],
    "w35": ["scripts/build_wave3.py", "w35"],
    "w36": ["scripts/build_wave3.py", "w36"],
    "w33": ["scripts/build_wave3.py", "w33"],
    "w34-skip-l4": [
        "scripts/build_wave3.py",
        "w34",
        "--skip-l4",
        "--ifd-score-map",
        "datasets/raw/ifd_scores.jsonl",
    ],
    "base_diversity": ["scripts/base_diversity_baseline.py"],
    "w32": ["scripts/build_wave3.py", "w32"],
    "w34-full": [
        "scripts/build_wave3.py",
        "w34",
        "--ifd-score-map",
        "datasets/raw/ifd_scores.jsonl",
    ],
    "w22": ["scripts/build_wave2_api.py", "--only", "w22"],
}


def test_argv_exactness_every_whitelist_entry():
    assert set(WHITELIST) == set(EXPECTED_ARGV)
    for script_id, argv in EXPECTED_ARGV.items():
        assert list(WHITELIST[script_id].argv) == argv, script_id


def test_tier_counts_and_split():
    tiers = Counter(spec.tier for spec in WHITELIST.values())
    assert tiers == {"safe": 5, "slow": 3, "billed": 3}
    # the two w2 sub-entries: w21 is safe+dry-run, w22 is billed+no dry-run
    assert "--dry-run" in WHITELIST["w21"].argv
    assert "--dry-run" not in WHITELIST["w22"].argv


def test_modifies_tracked_files_per_entry():
    assert "kb/cards.jsonl" in WHITELIST["w23"].modifies
    assert "kb/CARDS_HASH.txt" in WHITELIST["w23"].modifies
    assert "datasets/DATASET_HASH.txt" in WHITELIST["w35"].modifies
    assert WHITELIST["w32"].modifies == ()


def test_script_aliases():
    assert resolve_script_id("kb_rebuild") == "build_wave1"
    assert resolve_script_id("build_wave1") == "build_wave1"
    assert resolve_script_id("w21") == "w21"
    assert resolve_script_id("nope") is None


# --------------------------------------------------------------------------- #
# validate_options / effective argv / display argv
# --------------------------------------------------------------------------- #

def test_validate_options_w32_limit_ranges():
    spec = WHITELIST["w32"]
    with pytest.raises(ValueError, match="required"):
        validate_options(spec, {})
    with pytest.raises(ValueError, match=">= 1"):
        validate_options(spec, {"limit": 0})
    with pytest.raises(ValueError, match="<= 500"):
        validate_options(spec, {"limit": 600})
    with pytest.raises(ValueError, match="must be a int"):
        validate_options(spec, {"limit": "abc"})
    with pytest.raises(ValueError, match="unknown option"):
        validate_options(spec, {"limit": 50, "bogus": 1})
    assert validate_options(spec, {"limit": 50}) == {"limit": 50}
    assert validate_options(spec, {"limit": "50"}) == {"limit": 50}


def test_effective_argv_w32_appends_limit():
    spec = WHITELIST["w32"]
    assert effective_argv(spec, {"limit": 50}) == [
        "scripts/build_wave3.py", "w32", "--limit", "50",
    ]
    with pytest.raises(ValueError):
        effective_argv(spec, {})


def test_display_argv_matches_table():
    assert display_argv(WHITELIST["w21"]) == [
        "scripts/build_wave2_api.py", "--only", "w21", "--dry-run",
    ]
    assert display_argv(WHITELIST["w22"]) == [
        "scripts/build_wave2_api.py", "--only", "w22",
    ]
    assert display_argv(WHITELIST["w32"]) == [
        "scripts/build_wave3.py", "w32", "--limit", "<required>",
    ]
    assert display_argv(WHITELIST["w34-skip-l4"]) == list(WHITELIST["w34-skip-l4"].argv)


# --------------------------------------------------------------------------- #
# preflight checks
# --------------------------------------------------------------------------- #

def test_preflight_path_refusal_names_exact_missing_path(tmp_path):
    refusal = preflight_refusal(WHITELIST["w36"], root=tmp_path)
    assert refusal == (
        "preflight failed: data/vietnamese/Tarot-Vietnamese-API/data.txt is missing"
    )
    # a present path clears it
    p = tmp_path / "data/vietnamese/Tarot-Vietnamese-API"
    p.mkdir(parents=True)
    (p / "data.txt").write_text("x", encoding="utf-8")
    assert preflight_refusal(WHITELIST["w36"], root=tmp_path) is None


def test_preflight_env_checks_injected_env(tmp_path):
    from pathlib import Path

    spec = WHITELIST["w32"]
    root = Path("/nonexistent")
    ok = preflight_checks(spec, root=root,
                          env={"LLM_BASE_URL": "x", "LLM_API_KEY": "y", "LLM_MODEL": "z"})
    assert all(r.ok for r in ok)
    missing = preflight_checks(spec, root=root, env={})
    assert not any(r.ok for r in missing)
    assert any("LLM_BASE_URL" in r.reason for r in missing)


def test_preflight_cuda_injected(tmp_path):
    spec = WHITELIST["base_diversity"]
    raw = tmp_path / "datasets" / "raw"
    raw.mkdir(parents=True)
    (raw / "generated.jsonl").write_text("{}\n", encoding="utf-8")
    results = preflight_checks(spec, root=tmp_path, cuda_check=lambda: False)
    assert [r.ok for r in results] == [True, False]


def test_preflight_hf_model_injected(monkeypatch, tmp_path):
    spec = WHITELIST["w34-skip-l4"]
    monkeypatch.setattr(
        "tfvn.webapp.runs_gates._hf_model_present",
        lambda model: model == "Qwen/Qwen3-1.7B",
    )
    results = preflight_checks(spec, root=tmp_path)
    assert results[0].ok is True  # hf model present
    assert results[1].ok is False  # datasets/raw/ifd_scores.jsonl missing


# --------------------------------------------------------------------------- #
# gate_derivation (per-script from the A.5 table)
# --------------------------------------------------------------------------- #

def test_gate_derivation_exit_kind(fake_root):
    spec = WHITELIST["build_wave1"]
    assert gate_derivation(spec, 0, fake_root) == (True, "exit code 0")
    ok, detail = gate_derivation(spec, 3, fake_root)
    assert ok is False and detail == "exit code 3"


def test_gate_derivation_report_acceptance_failing_despite_exit_0(fake_root, tmp_path):
    spec = WHITELIST["w33"]  # report_acceptance over datasets/ablation_report.json
    ok, _ = gate_derivation(spec, 0, fake_root)
    assert ok is True  # fixture ablation_report has a truthy acceptance dict
    # a failing acceptance block -> gates_passed False EVEN THOUGH exit was 0
    alt = tmp_path / "alt"
    other = alt / "datasets"
    other.mkdir(parents=True)
    (other / "ablation_report.json").write_text(
        json.dumps({"acceptance": {"ablation_ok": False}}), encoding="utf-8"
    )
    ok, detail = gate_derivation(spec, 0, alt)
    assert ok is False
    assert "ablation_ok" in detail
    # missing report -> False with a "missing" detail, never a crash
    ok, detail = gate_derivation(spec, 0, tmp_path / "empty")
    assert ok is False and "missing" in detail


def test_gate_derivation_w22_aggregate(fake_root, tmp_path):
    spec = WHITELIST["w22"]  # w22_aggregate over kb/w2_2_gate_report.json
    ok, detail = gate_derivation(spec, 0, fake_root)
    assert ok is True
    assert "failed_gate=0" in detail and "0.95" in detail
    sub = tmp_path / "alt" / "kb"
    sub.mkdir(parents=True)
    # failed_gate == 1 -> gates_passed false
    (sub / "w2_2_gate_report.json").write_text(
        json.dumps({"aggregate": {"failed_gate": 1},
                    "negative_control_rejection_rate": 0.95}),
        encoding="utf-8",
    )
    assert gate_derivation(spec, 0, tmp_path / "alt")[0] is False
    # rejection below the 0.8 floor -> false
    (sub / "w2_2_gate_report.json").write_text(
        json.dumps({"aggregate": {"failed_gate": 0},
                    "negative_control_rejection_rate": 0.5}),
        encoding="utf-8",
    )
    ok, detail = gate_derivation(spec, 0, tmp_path / "alt")
    assert ok is False and "0.5" in detail
    # missing report -> false
    assert gate_derivation(spec, 0, tmp_path / "alt2")[0] is False


def test_gate_derivation_unknown_kind(tmp_path):
    from tfvn.webapp.runs_whitelist import RunSpec

    bogus = RunSpec(script_id="x", label="x", description="x", tier="safe",
                    argv=("x.py",), gate_kind="not_a_gate")
    ok, detail = gate_derivation(bogus, 0, tmp_path)
    assert ok is False and "unknown gate_kind" in detail


# --------------------------------------------------------------------------- #
# Fake runner + endpoint tests
# --------------------------------------------------------------------------- #
#
# The endpoints are async and the run finalises in a background task. They are
# therefore exercised by calling the route handlers directly under
# ``asyncio.run`` — a TestClient portal cancels the background task when the
# per-request portal closes, which would orphan every run mid-lifecycle.

class _FakeProc:
    def __init__(self):
        self.returncode = None

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


class FakeRunner:
    """Injected ``run_process``: records argv, streams emit lines, and holds
    the subprocess "alive" until the test releases it (thread-safe event)."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.release = threading.Event()
        self.emit_lines = ["fake line one", "fake line two"]
        self.exit_code = 0
        self.timed_out = False

    async def run(self, argv, *, cwd, env, emit, timeout_s=None, kill_grace=5.0,
                  on_start=None):
        self.calls.append((list(argv), cwd))
        if on_start is not None:
            on_start(_FakeProc())
        for line in self.emit_lines:
            emit(line)
        while not self.release.is_set():
            await asyncio.sleep(0.005)
        return self.exit_code, self.timed_out


@pytest.fixture
def runs_ctx(tmp_path, monkeypatch, fake_root):
    """runs.REPO_ROOT + history path + run_process all redirected to the
    tmp_path fixture tree; module globals reset before and after."""
    monkeypatch.setattr(runs_whitelist, "REPO_ROOT", fake_root)
    monkeypatch.setattr(runs_gates, "REPO_ROOT", fake_root)
    monkeypatch.setattr(runs, "REPO_ROOT", fake_root)
    monkeypatch.setattr(
        runs, "_history_path", lambda root=None: fake_root / "logs" / "webapp_runs.jsonl"
    )
    monkeypatch.setattr(runs_gates, "_cuda_available", lambda: False)  # no torch import
    # preflight_checks/preflight_refusal default root=REPO_ROOT at def time,
    # so the router-facing wrappers pin the fixture root explicitly.
    real_checks = runs.preflight_checks
    real_refusal = runs.preflight_refusal
    monkeypatch.setattr(
        runs, "preflight_checks", lambda spec: real_checks(spec, root=fake_root)
    )
    monkeypatch.setattr(
        runs, "preflight_refusal", lambda spec: real_refusal(spec, root=fake_root)
    )
    runner = FakeRunner()
    monkeypatch.setattr(runs, "run_process", runner.run)
    _reset_runs_state(runs)
    yield runs, fake_root, runner
    runner.release.set()
    deadline = time.time() + 8
    while runs._CURRENT is not None and time.time() < deadline:
        time.sleep(0.02)
    if runs._CURRENT is not None:
        task = runs._CURRENT.task
        if task is not None:
            task.cancel()
    _reset_runs_state(runs)


def _reset_runs_state(runs) -> None:
    runs._HISTORY = []
    runs._RECONCILED = False
    runs._ORPHANED = []
    runs._CURRENT = None
    runs._HANDLES = {}


async def _afinalized(runs, timeout=10.0) -> None:
    deadline = time.time() + timeout
    while runs._CURRENT is not None and time.time() < deadline:
        await asyncio.sleep(0.02)
    assert runs._CURRENT is None, "run never finalized"


def _run(coro) -> Any:
    return asyncio.run(coro)


def test_list_runs_whitelist_tiers_and_argv(runs_ctx):
    runs, _root, _runner = runs_ctx
    body = _run(runs.list_runs())
    assert len(body["scripts"]) == 11
    assert Counter(s["tier"] for s in body["scripts"]) == {"safe": 5, "slow": 3, "billed": 3}
    assert body["running"] is None and body["orphaned_pending"] == []
    w21 = next(s for s in body["scripts"] if s["script_id"] == "w21")
    assert "--dry-run" in w21["argv"]
    w22 = next(s for s in body["scripts"] if s["script_id"] == "w22")
    assert "--dry-run" not in w22["argv"]
    w32 = next(s for s in body["scripts"] if s["script_id"] == "w32")
    assert w32["argv"] == ["scripts/build_wave3.py", "w32", "--limit", "<required>"]
    assert w32["options"]["limit"]["required"] is True
    assert w32["timeout_s"] == 14400
    assert any(chip["label"] == "env LLM_BASE_URL set" for chip in w32["preflight"])


def test_billed_run_refuses_without_accept_cost(runs_ctx):
    from fastapi import HTTPException

    runs, _root, _runner = runs_ctx

    async def scenario():
        with pytest.raises(HTTPException) as exc:
            await runs.start_run("w22", runs.RunStartRequest(confirm=True))
        assert exc.value.status_code == 409
        assert "billed" in exc.value.detail and "accept_cost=true" in exc.value.detail
        with pytest.raises(HTTPException) as exc:
            await runs.start_run("w22", runs.RunStartRequest())
        assert exc.value.status_code == 409

    _run(scenario())


def test_slow_run_refuses_without_confirm(runs_ctx):
    from fastapi import HTTPException

    runs, _root, _runner = runs_ctx

    async def scenario():
        with pytest.raises(HTTPException) as exc:
            await runs.start_run("w33", runs.RunStartRequest())
        assert exc.value.status_code == 409
        assert "confirm=true" in exc.value.detail

    _run(scenario())


def test_w32_requires_explicit_limit(runs_ctx):
    from fastapi import HTTPException

    runs, _root, _runner = runs_ctx

    async def scenario():
        with pytest.raises(HTTPException) as exc:
            await runs.start_run(
                "w32", runs.RunStartRequest(confirm=True, accept_cost=True)
            )
        assert exc.value.status_code == 422
        assert "required" in exc.value.detail
        with pytest.raises(HTTPException) as exc:
            await runs.start_run("w32", runs.RunStartRequest(options={"limit": 600}))
        assert exc.value.status_code == 422

    _run(scenario())


def test_fresh_run_only_for_w32(runs_ctx):
    from fastapi import HTTPException

    runs, _root, _runner = runs_ctx

    async def scenario():
        with pytest.raises(HTTPException) as exc:
            await runs.start_run("w21", runs.RunStartRequest(fresh_run=True))
        assert exc.value.status_code == 422
        assert "fresh_run is only supported for w32" in exc.value.detail

    _run(scenario())


def test_preflight_refusal_blocks_run_with_exact_path(runs_ctx):
    from fastapi import HTTPException

    runs, _root, _runner = runs_ctx

    async def scenario():
        with pytest.raises(HTTPException) as exc:
            await runs.start_run("w36", runs.RunStartRequest())
        assert exc.value.status_code == 409
        assert exc.value.detail == (
            "preflight failed: data/vietnamese/Tarot-Vietnamese-API/data.txt is missing"
        )

    _run(scenario())


def test_unknown_script_404(runs_ctx):
    from fastapi import HTTPException

    runs, _root, _runner = runs_ctx

    async def scenario():
        with pytest.raises(HTTPException) as exc:
            await runs.start_run("nope", runs.RunStartRequest())
        assert exc.value.status_code == 404

    _run(scenario())


def test_single_flight_second_start_409(runs_ctx):
    from fastapi import HTTPException

    runs, _root, runner = runs_ctx

    async def scenario():
        r1 = await runs.start_run("w21", runs.RunStartRequest())
        assert r1["status"] == "running"
        with pytest.raises(HTTPException) as exc:
            await runs.start_run("w21", runs.RunStartRequest())
        assert exc.value.status_code == 409
        assert "a run is in progress" in exc.value.detail
        runner.release.set()
        await _afinalized(runs)

    _run(scenario())


def test_kill_terminates_and_marks_killed(runs_ctx):
    runs, _root, runner = runs_ctx

    async def scenario():
        r1 = await runs.start_run("w21", runs.RunStartRequest())
        run_id = r1["run_id"]
        for _ in range(500):
            if runs._HANDLES[run_id].proc is not None:
                break
            await asyncio.sleep(0.01)
        assert runs._HANDLES[run_id].proc is not None
        k = await runs.kill_run(run_id)
        assert k["killed"] is True
        runner.release.set()
        await _afinalized(runs)
        rec = (await runs.run_history())["runs"][0]
        assert rec["status"] == "killed" and rec["killed"] is True

    _run(scenario())


def test_fake_run_log_history_and_cache_invalidation(runs_ctx):
    from tfvn.webapp import catalog, stats

    runs, fake_root, runner = runs_ctx
    stats._cache_key = ("stale",)
    stats._cache_payload = {"stale": True}
    catalog._CACHE["key"] = "stale"
    catalog._CACHE["artifacts"] = []

    async def scenario():
        r1 = await runs.start_run("w21", runs.RunStartRequest())
        run_id = r1["run_id"]
        assert r1["argv"] == [
            "scripts/build_wave2_api.py", "--only", "w21", "--dry-run",
        ]
        # the background task emits synchronously once it starts; wait for it
        for _ in range(200):
            if runner.calls:
                break
            await asyncio.sleep(0.01)
        assert runner.calls
        assert (await runs.list_runs())["running"]["run_id"] == run_id
        log = await runs.run_log(run_id, 0)
        assert log["status"] == "running"
        assert log["lines"] == ["fake line one", "fake line two"]
        assert log["offset"] == 2 and log["truncated"] is False
        tail = await runs.run_log(run_id, 1)
        assert tail["lines"] == ["fake line two"]
        assert runner.calls[0][0] == [
            "scripts/build_wave2_api.py", "--only", "w21", "--dry-run",
        ]
        assert runner.calls[0][1] == str(fake_root)
        runner.release.set()
        await _afinalized(runs)
        assert (await runs.run_log(run_id))["status"] == "completed"
        hist = await runs.run_history()
        assert len(hist["runs"]) == 1
        rec = hist["runs"][0]
        assert rec["run_id"] == run_id
        assert rec["status"] == "completed" and rec["exit_code"] == 0
        assert rec["gates_passed"] is True and rec["gate_detail"] == "exit code 0"
        assert rec["killed"] is False and rec["timed_out"] is False
        assert rec["duration_s"] is not None
        hp = fake_root / "logs" / "webapp_runs.jsonl"
        assert hp.exists()
        assert json.loads(hp.read_text(encoding="utf-8").splitlines()[0])["run_id"] == run_id
        assert stats._cache_key is None and stats._cache_payload is None
        assert catalog._CACHE["key"] is None

    _run(scenario())


def test_boot_orphan_reconciliation_and_ack(runs_ctx):
    from fastapi import HTTPException

    runs, fake_root, _runner = runs_ctx
    rec = {
        "run_id": "run_orphan_0001",
        "script_id": "w21",
        "label": "W2.1 orientation attribution",
        "tier": "safe",
        "argv": ["scripts/build_wave2_api.py", "--only", "w21", "--dry-run"],
        "options": {},
        "status": "running",
        "exit_code": None,
        "gates_passed": None,
        "gate_detail": None,
        "killed": False,
        "timed_out": False,
        "started_at": "2026-08-12T10:00:00+07:00",
        "finished_at": None,
        "duration_s": None,
        "git_head": "abc123",
        "git_dirty": False,
        "modifies": ["kb/vn_orientation_attribution.json"],
        "acknowledged": False,
    }
    hp = fake_root / "logs"
    hp.mkdir(parents=True, exist_ok=True)
    from tfvn.serialise import dumps_canonical

    (hp / "webapp_runs.jsonl").write_text(dumps_canonical(rec) + "\n", encoding="utf-8")

    async def scenario():
        runs._load_history()  # patched _history_path -> the tmp file
        runs._RECONCILED = False
        listing = await runs.list_runs()
        assert listing["orphaned_pending"] == ["run_orphan_0001"]
        with pytest.raises(HTTPException) as exc:
            await runs.start_run("w21", runs.RunStartRequest())
        assert exc.value.status_code == 409
        assert "orphaned run is pending acknowledgement" in exc.value.detail
        hist = await runs.run_history()
        orphan = next(x for x in hist["runs"] if x["run_id"] == "run_orphan_0001")
        assert orphan["status"] == "orphaned" and orphan["acknowledged"] is False
        ack = await runs.ack_run("run_orphan_0001")
        assert ack["orphaned_pending"] == []
        assert (await runs.list_runs())["orphaned_pending"] == []
        with pytest.raises(HTTPException) as exc:
            await runs.ack_run("run_orphan_0001")
        assert exc.value.status_code == 409

    _run(scenario())
