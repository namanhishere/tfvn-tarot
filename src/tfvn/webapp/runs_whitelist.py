"""A.5 whitelist table — the single source of truth for pipeline re-runs.

One entry PER script invocation (argparse ``choices=`` take ONE value, so each
``--only`` value is its own entry). argv is fixed server-side; there is no
shell and no user-injected command string anywhere. Safe-tier entries are
genuinely offline: ``w21``/``w23`` MUST carry ``--dry-run`` (without it
``build_wave2_api.py:624-638`` unconditionally creates an ``LLMClient()`` and
calls the live ``GET {base}/models`` connection check); ``w22`` is the billed
reversed-synthesis entry and deliberately has no ``--dry-run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..llm_client import load_env

load_env()  # startup: subprocesses inherit .env keys, preflight env checks see them

REPO_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# Spec types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OptionSpec:
    """One CLI option (flag + value) for an entry's argv."""

    opt_type: str  # "int" | "float" | "str" | "bool"
    flag: str  # exact CLI flag, e.g. "--limit"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Any = None
    required: bool = False


@dataclass(frozen=True)
class PreflightSpec:
    """A single refuse-to-start check. ``kind`` drives the check function."""

    kind: str  # "path" | "env" | "hf_model" | "cuda"
    target: str  # relpath under REPO_ROOT / env var name / HF model id


@dataclass(frozen=True)
class RunSpec:
    script_id: str
    label: str
    description: str
    tier: str  # "safe" | "slow" | "billed"
    argv: tuple[str, ...]  # base argv; option flags appended at start time
    options: dict[str, OptionSpec] = field(default_factory=dict)
    timeout_s: int = 3600
    confirm_required: bool = False
    preflight: tuple[PreflightSpec, ...] = ()
    modifies: tuple[str, ...] = ()  # TRACKED files the run rewrites (confirm UI)
    gate_kind: str = "exit"  # "exit" | "report_acceptance" | "w22_aggregate"
    gate_report: Optional[str] = None  # relpath read for report-based gates


# --------------------------------------------------------------------------- #
# The whitelist
# --------------------------------------------------------------------------- #
# Gate derivation per script (NEVER the bare exit code):
#   exit             -> gates_passed = (exit_code == 0)
#   report_acceptance-> parse <gate_report>.acceptance, all flags truthy
#   w22_aggregate    -> kb/w2_2_gate_report.json has NO "acceptance" key;
#                       derive from aggregate.failed_gate == 0 AND
#                       negative_control_rejection_rate >= 0.8

WHITELIST: dict[str, RunSpec] = {
    # ------------------------------------------------------------------ safe --
    "build_wave1": RunSpec(
        script_id="build_wave1",
        label="Build Wave 1 KB",
        description="Regenerate all Wave 1 KB artifacts from the frozen source "
        "corpus (english_spine, vn_upright, spreads, register profile, compact "
        "cards, alias table). Fully offline.",
        tier="safe",
        argv=("scripts/build_wave1.py",),
        preflight=(
            PreflightSpec("path", "data/MANIFEST.md"),
            PreflightSpec("path", "data/CORPUS_HASH.txt"),
        ),
        modifies=(
            "kb/english_spine.jsonl",
            "kb/vn_upright.jsonl",
            "kb/spreads.jsonl",
            "kb/spreads_discrimination_report.json",
            "kb/vn_register_profile.json",
            "kb/english_spine.canonical.json",
            "kb/compact_cards.jsonl",
            "kb/card_name_whitelist.json",
            "kb/alias_table.json",
        ),
        gate_kind="exit",
    ),
    "w21": RunSpec(
        script_id="w21",
        label="W2.1 orientation attribution",
        description="Deterministic local polarity-lexicon proxy over the "
        "identical-field attribution (no API client). Writes "
        "kb/vn_orientation_attribution.json.",
        tier="safe",
        argv=("scripts/build_wave2_api.py", "--only", "w21", "--dry-run"),
        modifies=("kb/vn_orientation_attribution.json",),
        gate_kind="exit",
    ),
    "w23": RunSpec(
        script_id="w23",
        label="W2.3 bilingual KB freeze",
        description="Join english_spine + vn_upright + attribution + vn_spine "
        "into the 156-row kb/cards.jsonl and recompute kb/CARDS_HASH.txt. "
        "Offline --dry-run.",
        tier="safe",
        argv=("scripts/build_wave2_api.py", "--only", "w23", "--dry-run"),
        modifies=("kb/cards.jsonl", "kb/CARDS_HASH.txt"),
        gate_kind="exit",
    ),
    "w35": RunSpec(
        script_id="w35",
        label="W3.5 dedup cascade + coverage + DATASET_HASH",
        description="Analysis-only dedup cascade, coverage universe check and "
        "the canonical DATASET_HASH write over the shipped tiers. Never "
        "rewrites the tiers themselves.",
        tier="safe",
        argv=("scripts/build_wave3.py", "w35"),
        modifies=("datasets/coverage_report.json", "datasets/DATASET_HASH.txt"),
        gate_kind="report_acceptance",
        gate_report="datasets/coverage_report.json",
    ),
    "w36": RunSpec(
        script_id="w36",
        label="W3.6 stratified splits + anchor isolation",
        description="85/8/7 stratified splits over task_type/length_band/"
        "register units plus the 30-row anchor set from the frozen Vietnamese "
        "source corpus.",
        tier="safe",
        argv=("scripts/build_wave3.py", "w36"),
        preflight=(PreflightSpec("path", "data/vietnamese/Tarot-Vietnamese-API/data.txt"),),
        modifies=(
            "datasets/anchor/anchor_readings.jsonl",
            "datasets/splits.json",
            "datasets/split_stats.json",
        ),
        gate_kind="report_acceptance",
        gate_report="datasets/split_stats.json",
    ),
    # ------------------------------------------------------------------ slow --
    "w33": RunSpec(
        script_id="w33",
        label="W3.3 anti-collapse ablation + diversity",
        description="Anti-collapse ablation and corpus diversity metrics over "
        "datasets/raw/generated.jsonl. Local compute only.",
        tier="slow",
        argv=("scripts/build_wave3.py", "w33"),
        preflight=(PreflightSpec("path", "datasets/raw/generated.jsonl"),),
        modifies=("datasets/ablation_report.json",),
        timeout_s=7200,
        confirm_required=True,
        gate_kind="report_acceptance",
        gate_report="datasets/ablation_report.json",
    ),
    "w34-skip-l4": RunSpec(
        script_id="w34-skip-l4",
        label="W3.4 filter without L4 judge",
        description="L1 programmatic + L2 IFD (loads local Qwen3-1.7B via torch) "
        "+ L3 Deita. Skips the billed L4 judge with --skip-l4; reuses "
        "datasets/raw/ifd_scores.jsonl.",
        tier="slow",
        argv=(
            "scripts/build_wave3.py",
            "w34",
            "--skip-l4",
            "--ifd-score-map",
            "datasets/raw/ifd_scores.jsonl",
        ),
        preflight=(
            PreflightSpec("hf_model", "Qwen/Qwen3-1.7B"),
            PreflightSpec("path", "datasets/raw/ifd_scores.jsonl"),
        ),
        modifies=(
            "datasets/filtered_core.jsonl",
            "datasets/filtered_bulk.jsonl",
            "datasets/filter_report.json",
        ),
        timeout_s=7200,
        confirm_required=True,
        gate_kind="report_acceptance",
        gate_report="datasets/filter_report.json",
    ),
    "base_diversity": RunSpec(
        script_id="base_diversity",
        label="Base-model diversity baseline",
        description="GPU/torch job: 200 Vietnamese readings from Qwen3-1.7B on "
        "the corpus's own prompts; calibrates the distinct-2 floor. Rewrites "
        "datasets/base_diversity_baseline.json.",
        tier="slow",
        argv=("scripts/base_diversity_baseline.py",),
        preflight=(
            PreflightSpec("path", "datasets/raw/generated.jsonl"),
            PreflightSpec("cuda", ""),
        ),
        modifies=("datasets/base_diversity_baseline.json",),
        timeout_s=7200,
        confirm_required=True,
        gate_kind="exit",
    ),
    # ---------------------------------------------------------------- billed --
    "w32": RunSpec(
        script_id="w32",
        label="W3.2 generation (billed)",
        description="LLM generation loop APPENDING to datasets/raw/generated.jsonl "
        "(never truncates). --limit is per-round; the script default of 40 is a "
        "smoke size so this entry REQUIRES an explicit limit. Optionally move the "
        "old generated.jsonl aside first (fresh_run).",
        tier="billed",
        argv=("scripts/build_wave3.py", "w32"),
        options={
            "limit": OptionSpec("int", "--limit", minimum=1, maximum=500, required=True),
        },
        preflight=(
            PreflightSpec("env", "LLM_BASE_URL"),
            PreflightSpec("env", "LLM_API_KEY"),
            PreflightSpec("env", "LLM_MODEL"),
        ),
        modifies=(),  # appends gitignored datasets/raw/generated.jsonl
        timeout_s=14400,
        confirm_required=True,
        gate_kind="exit",
    ),
    "w34-full": RunSpec(
        script_id="w34-full",
        label="W3.4 full filter incl. L4 judge (billed)",
        description="All four layers; L4 runs the calibrated independent judge "
        "on LLM_MODEL_SONNET. Reuses datasets/raw/ifd_scores.jsonl for L2.",
        tier="billed",
        argv=(
            "scripts/build_wave3.py",
            "w34",
            "--ifd-score-map",
            "datasets/raw/ifd_scores.jsonl",
        ),
        preflight=(PreflightSpec("env", "LLM_MODEL_SONNET"),),
        modifies=(
            "datasets/filtered_core.jsonl",
            "datasets/filtered_bulk.jsonl",
            "datasets/filter_report.json",
        ),
        timeout_s=14400,
        confirm_required=True,
        gate_kind="report_acceptance",
        gate_report="datasets/filter_report.json",
    ),
    "w22": RunSpec(
        script_id="w22",
        label="W2.2 reversed synthesis (billed)",
        description="Generate Vietnamese reversed meanings for all 78 cards via "
        "the LLM (default --neg-control 20 adds ~40 extra API calls). Writes "
        "kb/vn_spine.jsonl + kb/w2_2_gate_report.json.",
        tier="billed",
        argv=("scripts/build_wave2_api.py", "--only", "w22"),
        preflight=(
            PreflightSpec("env", "LLM_BASE_URL"),
            PreflightSpec("env", "LLM_API_KEY"),
            PreflightSpec("env", "LLM_MODEL"),
        ),
        modifies=("kb/vn_spine.jsonl", "kb/w2_2_gate_report.json"),
        timeout_s=14400,
        confirm_required=True,
        gate_kind="w22_aggregate",
        gate_report="kb/w2_2_gate_report.json",
    ),
}

# Scripts that must NEVER be runnable from the webapp (system install etc).
EXCLUDED_SCRIPTS: tuple[str, ...] = ("scripts/install_cmake_llama.sh",)

# Friendly ids accepted by POST /api/runs/{script_id} that resolve to a
# canonical WHITELIST entry (the listing always shows the canonical id).
SCRIPT_ALIASES: dict[str, str] = {
    "kb_rebuild": "build_wave1",
}


def resolve_script_id(script_id: str) -> Optional[str]:
    """Canonical WHITELIST key for a user-supplied id (aliases allowed)."""
    if script_id in WHITELIST:
        return script_id
    return SCRIPT_ALIASES.get(script_id)
