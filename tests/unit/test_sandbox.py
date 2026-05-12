"""Unit tests for `panoptes.sandbox.python_exec`."""

from __future__ import annotations

import sys

import pytest

from panoptes.sandbox.python_exec import (
    SandboxLimits,
    humaneval_check,
    run_python,
)


def test_passing_code_returns_passed_true() -> None:
    result = run_python("x = 1 + 1\nassert x == 2\n")
    assert result.passed
    assert result.returncode == 0
    assert not result.timed_out


def test_failing_assertion_returns_passed_false() -> None:
    result = run_python("assert 1 == 2, 'nope'\n")
    assert not result.passed
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


@pytest.mark.skipif(
    sys.platform == "win32", reason="preexec_fn / setrlimit unavailable on Windows"
)
def test_timeout_kills_long_running_code() -> None:
    code = "import time\nwhile True:\n    time.sleep(1)\n"
    result = run_python(code, limits=SandboxLimits(cpu_seconds=2, memory_mb=128, wall_seconds=1.0))
    assert not result.passed
    assert result.timed_out
    assert result.duration_s < 5.0


def test_humaneval_check_passes_on_correct_solution() -> None:
    prompt = (
        "def add(a, b):\n"
        '    """Return the sum."""\n'
    )
    candidate = "    return a + b\n"
    test = "def check(fn):\n    assert fn(2, 3) == 5\n    assert fn(0, 0) == 0\n"
    result = humaneval_check(prompt, candidate, test, "add")
    assert result.passed


def test_humaneval_check_fails_on_wrong_solution() -> None:
    prompt = "def add(a, b):\n    pass\n"
    candidate = "    return a - b\n"
    test = "def check(fn):\n    assert fn(2, 3) == 5\n"
    result = humaneval_check(prompt, candidate, test, "add")
    assert not result.passed
