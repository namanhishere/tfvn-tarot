"""Whitelisted pipeline script runner with tiers, logs and history (todo A.5)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Placeholder — A.5 owns the whitelist table (safe/slow/billed tiers),
# asyncio subprocess runner, streaming log scrubber, single-flight lock,
# kill and run history in logs/webapp_runs.jsonl.
