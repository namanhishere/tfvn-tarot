"""A.5 async subprocess runner core — streaming, scrubber, ring buffer, kill.

Deliberately free of HTTP and server state: :func:`run_process` takes the
command list as a plain argument, so pytest (todo C.1) can inject fake
commands. Never uses a shell and never blocks on a PIPE'd process — stdout is
drained line-by-line as it arrives, avoiding the 64 KB pipe-buffer deadlock.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from collections import deque
from typing import Callable, Mapping, Optional

# --------------------------------------------------------------------------- #
# Log scrubber — .env secrets never reach the UI or log files
# --------------------------------------------------------------------------- #

_SK_RE = re.compile(r"sk-[A-Za-z0-9]{8,}")
_BEARER_RE = re.compile(r"Bearer\s+\S+")
_SECRET_VAR_RE = re.compile(r"(^|_)(KEY|TOKEN)(_|$)", re.IGNORECASE)


def build_scrubber(
    env: Optional[Mapping[str, str]] = None,
) -> Callable[[str], str]:
    """Mask ``sk-…``, ``Bearer …`` and the EXACT value of every ``*_KEY`` /
    ``*_TOKEN`` env var (regexes alone miss ``sk-ant-…``-style keys whose
    values contain internal dashes). Returns a line -> scrubbed-line fn."""
    env = os.environ if env is None else env
    secrets = sorted(
        {
            value
            for key, value in env.items()
            if value and _SECRET_VAR_RE.search(key) and len(value) >= 8
        },
        key=len,
        reverse=True,  # longest first so shorter secrets can't corrupt longer ones
    )

    def scrub(line: str) -> str:
        line = _SK_RE.sub("***", line)
        line = _BEARER_RE.sub("***", line)
        for secret in secrets:
            line = line.replace(secret, "***")
        return line

    return scrub


# --------------------------------------------------------------------------- #
# Ring buffer for live log tails
# --------------------------------------------------------------------------- #


class RingLog:
    """Bounded line ring buffer (last ``capacity`` lines) with global offsets.

    ``append`` returns the global index of the stored line; ``read(offset)``
    serves the tail starting at ``offset`` and reports ``truncated`` when the
    requested offset predates the retained window.
    """

    def __init__(self, capacity: int = 5000) -> None:
        self._buf: deque[str] = deque(maxlen=capacity)
        self._start = 0  # global index of self._buf[0]
        self._total = 0  # lines ever appended

    @property
    def total(self) -> int:
        return self._total

    def append(self, line: str) -> int:
        if len(self._buf) >= self._buf.maxlen:
            self._start += 1
        self._buf.append(line)
        self._total += 1
        return self._total - 1

    def read(self, offset: int = 0) -> tuple[list[str], int, bool]:
        """Return (lines, next_offset, truncated) for the tail after ``offset``."""
        if offset < 0:
            offset = 0
        if offset >= self._total:
            return [], self._total, False
        truncated = offset < self._start
        start = max(offset, self._start)
        lines = list(self._buf)[start - self._start :]
        return lines, self._total, truncated


# --------------------------------------------------------------------------- #
# Subprocess lifecycle
# --------------------------------------------------------------------------- #


class RunTimeout(Exception):
    """Raised internally when a run exceeds its per-tier timeout."""


async def _terminate_graceful(
    proc: asyncio.subprocess.Process, grace: float
) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


async def run_process(
    argv: list[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    emit: Callable[[str], None],
    timeout_s: Optional[float] = None,
    kill_grace: float = 5.0,
    on_start: Optional[Callable[[asyncio.subprocess.Process], None]] = None,
) -> tuple[int, bool]:
    """Run ``argv`` (no shell), streaming scrubbed stdout lines to ``emit``.

    Returns ``(returncode, timed_out)``. ``emit`` MUST be exception-safe
    (a raise here would leak the process); the orchestrator's emit only
    touches in-memory buffers and the run log file.

    Repo scripts carry no ``+x`` bit (mode 644, always run as
    ``python3 scripts/...``), so a ``*.py`` first argument is executed via
    ``sys.executable`` — the whitelist argv stays exactly as declared.
    """
    exec_argv = list(argv)
    if exec_argv and exec_argv[0].endswith(".py"):
        exec_argv = [sys.executable, *exec_argv]
    proc = await asyncio.create_subprocess_exec(
        *exec_argv,
        cwd=cwd,
        env={**env},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if on_start is not None:
        on_start(proc)
    assert proc.stdout is not None
    loop = asyncio.get_event_loop()
    deadline = (loop.time() + timeout_s) if timeout_s else None
    timed_out = False
    try:
        while True:
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    timed_out = True
                    break
                line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=remaining
                )
            else:
                line = await proc.stdout.readline()
            if not line:
                break
            emit(line.decode("utf-8", errors="replace").rstrip("\n"))
    except asyncio.TimeoutError:
        timed_out = True
    if timed_out:
        await _terminate_graceful(proc, kill_grace)
    return await proc.wait(), timed_out


async def kill_process(
    proc: Optional[asyncio.subprocess.Process], grace: float = 5.0
) -> bool:
    """Terminate then kill after ``grace`` seconds. False if nothing to kill."""
    if proc is None or proc.returncode is not None:
        return False
    await _terminate_graceful(proc, grace)
    return True
