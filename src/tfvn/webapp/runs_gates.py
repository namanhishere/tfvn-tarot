"""A.5 argv assembly, preflight checks and gate derivation.

Pure functions over the whitelist table — no HTTP, no subprocesses, no state —
so C.1 can unit-test them directly (exact argv, option ranges, preflight
refusal naming the exact missing path, gate derivation from fixture reports).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .runs_whitelist import REPO_ROOT, RunSpec

# --------------------------------------------------------------------------- #
# argv assembly (never a shell, never a user-injected command string)
# --------------------------------------------------------------------------- #


def validate_options(spec: RunSpec, options: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce + range-check option values. Raises ValueError on any problem."""
    unknown = [k for k in options if k not in spec.options]
    if unknown:
        raise ValueError(
            f"unknown option(s) for {spec.script_id}: {', '.join(sorted(unknown))}"
        )
    out: dict[str, Any] = {}
    for key, o in spec.options.items():
        raw = options.get(key, o.default)
        if raw is None:
            if o.required:
                raise ValueError(f"option {key!r} is required for {spec.script_id}")
            continue
        try:
            if o.opt_type == "int":
                val: Any = int(raw)
            elif o.opt_type == "float":
                val = float(raw)
            elif o.opt_type == "bool":
                val = bool(raw)
            else:
                val = str(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"option {key!r} must be a {o.opt_type} (got {raw!r})"
            ) from None
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if o.minimum is not None and val < o.minimum:
                raise ValueError(f"option {key!r} must be >= {o.minimum} (got {val})")
            if o.maximum is not None and val > o.maximum:
                raise ValueError(f"option {key!r} must be <= {o.maximum} (got {val})")
        out[key] = val
    return out


def effective_argv(
    spec: RunSpec, options: Optional[Mapping[str, Any]] = None
) -> list[str]:
    """Concrete argv for a run: table argv + validated option flags."""
    opts = validate_options(spec, options or {})
    argv = list(spec.argv)
    for key, o in spec.options.items():
        if key not in opts:
            continue
        if o.opt_type == "bool":
            if opts[key]:
                argv.append(o.flag)
        else:
            argv.extend([o.flag, str(opts[key])])
    return argv


def display_argv(spec: RunSpec) -> list[str]:
    """Read-only effective-argv display for GET /api/runs.

    Required options without a default (w32 ``--limit``) render as
    ``<required>`` so the UI never implies a silent default is used.
    """
    argv = list(spec.argv)
    for key, o in spec.options.items():
        if o.required and o.default is None:
            argv.extend([o.flag, "<required>"])
        elif o.default is not None:
            if o.opt_type == "bool":
                if o.default:
                    argv.append(o.flag)
            else:
                argv.extend([o.flag, str(o.default)])
    return argv


# --------------------------------------------------------------------------- #
# Preflight checks (refuse a run whose inputs are missing — exact path named)
# --------------------------------------------------------------------------- #


@dataclass
class PreflightResult:
    ok: bool
    label: str
    reason: Optional[str] = None


def _hf_model_present(model_id: str) -> bool:
    dirname = "models--" + model_id.replace("/", "--")
    snap = Path.home() / ".cache" / "huggingface" / "hub" / dirname / "snapshots"
    try:
        return snap.is_dir() and any(snap.iterdir())
    except OSError:
        return False


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def preflight_checks(
    spec: RunSpec,
    root: Path = REPO_ROOT,
    env: Optional[Mapping[str, str]] = None,
    cuda_check: Optional[Callable[[], bool]] = None,
) -> list[PreflightResult]:
    env = os.environ if env is None else env
    cuda = _cuda_available if cuda_check is None else cuda_check
    results: list[PreflightResult] = []
    for p in spec.preflight:
        if p.kind == "path":
            path = root / p.target
            ok = path.exists()
            results.append(
                PreflightResult(
                    ok,
                    f"{p.target} exists",
                    None if ok else f"{p.target} is missing",
                )
            )
        elif p.kind == "env":
            ok = bool(env.get(p.target, "").strip())
            results.append(
                PreflightResult(
                    ok,
                    f"env {p.target} set",
                    None if ok else f"env {p.target} is not set",
                )
            )
        elif p.kind == "hf_model":
            ok = _hf_model_present(p.target)
            results.append(
                PreflightResult(
                    ok,
                    f"{p.target} in HF cache",
                    None
                    if ok
                    else f"{p.target} not found in HF cache "
                    "(~/.cache/huggingface/hub/models--{p.target.replace('/', '--')})",
                )
            )
        elif p.kind == "cuda":
            ok = cuda()
            results.append(
                PreflightResult(
                    ok,
                    "CUDA available",
                    None if ok else "CUDA not available (base_diversity needs a GPU)",
                )
            )
    return results


def preflight_refusal(
    spec: RunSpec,
    root: Path = REPO_ROOT,
    env: Optional[Mapping[str, str]] = None,
    cuda_check: Optional[Callable[[], bool]] = None,
) -> Optional[str]:
    """None when the entry passes preflight; else a refusal naming every failure."""
    failed = [r for r in preflight_checks(spec, root, env, cuda_check) if not r.ok]
    if not failed:
        return None
    return "preflight failed: " + "; ".join(r.reason or r.label for r in failed)


# --------------------------------------------------------------------------- #
# Gate derivation — the exit code is NEVER the gate result.
# --------------------------------------------------------------------------- #


def _read_report(
    root: Path, rel: str
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    path = root / rel
    if not path.exists():
        return None, f"{rel} missing (gate cannot be evaluated)"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{rel} unreadable: {exc}"
    if not isinstance(report, dict):
        return None, f"{rel} is not a JSON object"
    return report, None


def gate_derivation(
    spec: RunSpec, exit_code: int, root: Path = REPO_ROOT
) -> tuple[bool, str]:
    """(gates_passed, human-readable detail) for a finished run."""
    if spec.gate_kind == "exit":
        return exit_code == 0, f"exit code {exit_code}"
    if spec.gate_kind == "report_acceptance":
        report, err = _read_report(root, spec.gate_report or "")
        if err:
            return False, err
        acc = report.get("acceptance")  # type: ignore[union-attr]
        if not isinstance(acc, dict):
            return False, f"no 'acceptance' dict in {spec.gate_report}"
        failed = sorted(k for k, v in acc.items() if not v)
        if failed:
            return False, (
                f"{spec.gate_report} acceptance failing: {', '.join(failed)}"
            )
        return True, f"{spec.gate_report} acceptance: {len(acc)}/{len(acc)} pass"
    if spec.gate_kind == "w22_aggregate":
        report, err = _read_report(root, spec.gate_report or "")
        if err:
            return False, err
        # kb/w2_2_gate_report.json has NO "acceptance" key — derive from the
        # top-level aggregate + negative-control rejection rate.
        agg = report.get("aggregate") or {}
        failed = agg.get("failed_gate")
        rejection = report.get("negative_control_rejection_rate")
        ok = failed == 0 and rejection is not None and rejection >= 0.8
        detail = (
            f"aggregate.failed_gate={failed}, "
            f"negative_control_rejection_rate={rejection}"
        )
        return ok, detail
    return False, f"unknown gate_kind {spec.gate_kind!r} for {spec.script_id}"

