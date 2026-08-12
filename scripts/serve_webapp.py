#!/usr/bin/env python3
"""Serve the tfvn dataset viewer (FastAPI via uvicorn)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the tfvn dataset viewer")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to bind (default: 8000)",
    )
    args = parser.parse_args()

    # noqa: E402 — import after sys.path insert (repo convention)
    from tfvn.webapp.server import app  # noqa: E402

    # workers=1 and no --reload: reload would watch cwd and restart on
    # kb/datasets/logs writes, orphaning run subprocesses.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
