"""runs_runner core tests (A.5 / C.1).

Exercises the real :func:`run_process` with FAKE commands — small throwaway
scripts under ``tmp_path`` (never any ``build_wave*.py``), plus the RingLog
and log scrubber units.
"""

from __future__ import annotations

import asyncio
import os
import sys

from tfvn.webapp.runs_runner import (
    RingLog,
    build_scrubber,
    kill_process,
    run_process,
)


def _await(coro):
    return asyncio.run(coro)


def test_run_process_py_argv_interpreter_prefix(tmp_path):
    """A ``*.py`` argv[0] is executed via sys.executable (repo scripts are
    mode 644 and have no +x bit) — the declared argv stays intact."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "fake_script.py").write_text(
        "import sys\n"
        "sys.stdout.write('out line\\n'); sys.stdout.flush()\n"
        "sys.stderr.write('err line\\n')\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    lines: list[str] = []

    async def scenario():
        return await run_process(
            ["scripts/fake_script.py"],
            cwd=str(tmp_path),
            env={**os.environ},
            emit=lines.append,
        )

    code, timed_out = _await(scenario())
    assert code == 3 and timed_out is False
    # stderr merged into stdout, line-by-line, in order
    assert lines == ["out line", "err line"]


def test_run_process_non_py_argv_untouched(tmp_path):
    """argv[0] without a .py suffix is executed as declared (no interpreter
    prepend) — here sys.executable itself plus ``-c``."""
    lines: list[str] = []

    async def scenario():
        return await run_process(
            [sys.executable, "-c", "print('hello from -c')"],
            cwd=str(tmp_path),
            env={**os.environ},
            emit=lines.append,
        )

    code, timed_out = _await(scenario())
    assert code == 0 and timed_out is False
    assert lines == ["hello from -c"]


def test_run_process_timeout_terminates(tmp_path):
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    lines: list[str] = []

    async def scenario():
        return await run_process(
            ["sleeper.py"],
            cwd=str(tmp_path),
            env={**os.environ},
            emit=lines.append,
            timeout_s=0.2,
            kill_grace=2.0,
        )

    code, timed_out = _await(scenario())
    assert timed_out is True
    assert code == -15  # SIGTERM from the graceful terminate


def test_kill_process_sigterm(tmp_path):
    (tmp_path / "sleeper.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    async def scenario():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "sleeper.py",
            cwd=str(tmp_path),
            stdout=asyncio.subprocess.DEVNULL,
        )
        assert await kill_process(proc) is True
        return proc.returncode

    assert _await(scenario()) == -15
    assert _await(kill_process(None)) is False


def test_ringlog_bounded_capacity_and_offsets():
    ring = RingLog(capacity=3)
    for i in range(5):
        ring.append(f"line{i}")
    assert ring.total == 5
    lines, offset, truncated = ring.read(0)
    assert truncated is True
    assert lines == ["line2", "line3", "line4"] and offset == 5
    # offset exactly at the retained window start is NOT truncated
    lines, offset, truncated = ring.read(2)
    assert truncated is False and lines == ["line2", "line3", "line4"]
    lines, offset, truncated = ring.read(4)
    assert truncated is False and lines == ["line4"] and offset == 5
    lines, offset, truncated = ring.read(99)
    assert lines == [] and offset == 5 and truncated is False
    assert ring.read(-3) == (["line2", "line3", "line4"], 5, True)


def test_ringlog_append_returns_global_index():
    ring = RingLog(capacity=2)
    assert ring.append("a") == 0
    assert ring.append("b") == 1
    assert ring.append("c") == 2  # "a" evicted, global index still 2
    assert ring.total == 3


def test_scrubber_masks_sk_bearer_and_exact_env_values():
    scrub = build_scrubber(
        {
            "MY_API_KEY": "sk-ant-internal-dashes-12345678",
            "OTHER_TOKEN": "tok12345678secret",
        }
    )
    assert scrub("sk-abcdef1234567890") == "***"
    assert scrub("authorization: Bearer abc.def.ghi") == "authorization: ***"
    # regex misses the internal-dash key — exact-value masking catches it
    assert scrub("my key is sk-ant-internal-dashes-12345678") == "my key is ***"
    assert scrub("using tok12345678secret here") == "using *** here"
    # shortest env secret must not corrupt a longer one
    assert "tok12345678secret" not in scrub("tok12345678secret")


def test_scrubber_no_env_secrets_noop():
    scrub = build_scrubber({"SOME_VAR": "plain value"})
    assert scrub("plain line") == "plain line"
