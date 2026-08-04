"""Run the Wave 1 KB assertion suite as a pytest entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.assert_kb import run_assertions  # noqa: E402


def test_kb_assertion_suite():
    rc = run_assertions(ROOT / "kb")
    assert rc == 0
