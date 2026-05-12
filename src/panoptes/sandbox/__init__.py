"""Sandboxed code execution.

A `subprocess`-based Python sandbox for executing HumanEval / MBPP
candidate solutions against their test blocks. A Docker backend is the
obvious next step (better isolation, network deny by default) but adds
setup friction; the subprocess backend is enough for trusted users
running their own benchmarks locally.

Threat model: PANOPTES is *not* a hardened sandbox for adversarial code.
We bound CPU, memory, and wall time, and run in a temp directory, which
catches accidents and crashes from buggy LLM outputs. Do not run on
adversarial code without a real isolation layer (gVisor, Firecracker,
Docker rootless, etc.).
"""

from panoptes.sandbox.python_exec import (
    SandboxLimits,
    SandboxResult,
    run_python,
)

__all__ = ["SandboxLimits", "SandboxResult", "run_python"]
