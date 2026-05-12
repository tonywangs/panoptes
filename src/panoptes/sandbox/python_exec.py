"""Subprocess Python sandbox for HumanEval / MBPP candidate grading.

`run_python(code, *, limits)` writes `code` to a tempdir, invokes the
Python interpreter with `resource.setrlimit`-imposed CPU and memory caps,
and enforces a hard wall-clock timeout. Returns `SandboxResult(passed,
stdout, stderr, duration_s)`.

This is the standard "tempdir + setrlimit" recipe used in `human-eval`
and similar benchmarks. It is *not* hardened for adversarial code; see
the package docstring.

References
----------
- Chen et al. (2021), *Evaluating Large Language Models Trained on Code* §3.1
  (the HumanEval `check_correctness` reference implementation).
"""

from __future__ import annotations

import contextlib
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Resource caps for one execution.

    Defaults are conservative: 8s CPU, 512MB RAM, 10s wall. The wall
    timeout is `max(wall_s, cpu_s + 2)` to give the kernel a beat to
    enforce setrlimit before we kill the process ourselves.
    """

    cpu_seconds: int = 8
    memory_mb: int = 512
    wall_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Outcome of one sandboxed execution."""

    passed: bool
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool


def _setrlimits(limits: SandboxLimits) -> None:  # pragma: no cover - child only
    """Pre-exec hook: install CPU and memory caps in the child process.

    Called inside the forked subprocess before exec(). Runs in the child;
    coverage on this branch is intentional.
    """
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    mem_bytes = limits.memory_mb * 1024 * 1024
    # RLIMIT_AS bounds the process's virtual memory; macOS doesn't honor it
    # on all builds, but it's harmless when ignored.
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))


def run_python(
    code: str,
    *,
    limits: SandboxLimits | None = None,
    python_executable: str | None = None,
) -> SandboxResult:
    """Run `code` as a Python module under resource limits.

    Returns `passed=True` iff the subprocess exited with status 0 inside
    the wall-clock budget. `timed_out=True` means we killed the child
    after `limits.wall_seconds` elapsed.
    """
    eff_limits = limits if limits is not None else SandboxLimits()
    interpreter = python_executable if python_executable is not None else sys.executable

    workdir = Path(tempfile.mkdtemp(prefix="panoptes-sandbox-"))
    script = workdir / "candidate.py"
    script.write_text(code, encoding="utf-8")

    wall = max(eff_limits.wall_seconds, float(eff_limits.cpu_seconds) + 2.0)
    t0 = time.perf_counter()
    timed_out = False
    stdout = ""
    stderr = ""
    returncode = -1
    passed = False
    try:
        proc = subprocess.run(
            [interpreter, str(script)],
            capture_output=True,
            text=True,
            cwd=workdir,
            timeout=wall,
            preexec_fn=lambda: _setrlimits(eff_limits),
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
        passed = returncode == 0
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        # `text=True` above means stdout/stderr are str | None; coerce just in case.
        stdout_obj = exc.stdout
        if isinstance(stdout_obj, (bytes, bytearray)):
            stdout = bytes(stdout_obj).decode("utf-8", errors="replace")
        elif stdout_obj is not None:
            stdout = str(stdout_obj)
        stderr_obj = exc.stderr
        if isinstance(stderr_obj, (bytes, bytearray)):
            stderr = bytes(stderr_obj).decode("utf-8", errors="replace")
        elif stderr_obj is not None:
            stderr = str(stderr_obj)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    duration = time.perf_counter() - t0
    return SandboxResult(
        passed=passed,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration,
        timed_out=timed_out,
    )


def humaneval_check(prompt: str, candidate: str, test: str, entry_point: str) -> SandboxResult:
    """Convenience: run the HumanEval `check_correctness` pattern.

    Builds `prompt + candidate + test + check(entry_point)` as the script,
    where `test` defines a top-level `check(candidate)` function. Returns
    the sandbox result; `passed` corresponds to all test assertions holding.
    """
    full_code = f"{prompt}\n{candidate}\n{test}\ncheck({entry_point})\n"
    return run_python(full_code)
