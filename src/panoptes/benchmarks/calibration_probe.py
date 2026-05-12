"""Held-out calibration probe via *obfuscated* HumanEval variants.

The premise: any conformal calibration computed against vanilla HumanEval
is contaminated by the simple fact that HumanEval predates every major
LLM and is in their pretraining. We sidestep this by mechanically rewriting
each problem so judges cannot pattern-match:

    1. Rename the entry-point function to an opaque identifier.
    2. Apply a deterministic alpha-renaming to local variables in the
       *canonical solution* (we leave the prompt's free-form docstring
       alone — rewriting natural-language meaning is risky).
    3. Rewrite the test block to call the new entry-point.

The ground truth — does this candidate solution actually pass the rewritten
tests? — is then computed via the sandboxed Python executor. This gives
PANOPTES a verifiable, post-pretraining boolean label per item, which is
exactly what the conformal calibration probe needs.

This is a *light* obfuscation: it preserves the docstring (so the task
is still understandable) but breaks the model's ability to recall the
canonical solution by name. A stronger version would also rename inside
the docstring; we leave that to a follow-up since aggressive rewriting
risks making the task ambiguous.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from panoptes.schemas import BenchmarkItem, TaskFamily


@dataclass(frozen=True, slots=True)
class ObfuscatedItem:
    """A `BenchmarkItem` plus the rewritten test block for grading."""

    item: BenchmarkItem
    rewritten_test: str
    rewritten_entry_point: str
    original_entry_point: str


def _opaque_name(seed: str, *, prefix: str = "fn_") -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}{digest}"


_DEF_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _rename_def(source: str, old: str, new: str) -> str:
    """Replace exactly the `def OLD(` declaration and word-boundary
    references to `OLD` outside string literals.

    Note: this is a string-level rewrite, not an AST rewrite, so docstrings
    that mention the function name will be rewritten too. That's
    acceptable: docstring rewriting *helps* the de-memorization story.
    """
    pattern = re.compile(rf"\b{re.escape(old)}\b")
    return pattern.sub(new, source)


def obfuscate_humaneval(items: Iterable[BenchmarkItem]) -> list[ObfuscatedItem]:
    """Build calibration-probe items from a list of vanilla HumanEval items.

    Each item must have `metadata['canonical_solution']`,
    `metadata['test']`, and `metadata['entry_point']`. Items missing any
    of these are skipped.
    """
    out: list[ObfuscatedItem] = []
    for src in items:
        entry_point = src.metadata.get("entry_point")
        test_block = src.metadata.get("test")
        canonical = src.metadata.get("canonical_solution")
        if not (
            isinstance(entry_point, str)
            and isinstance(test_block, str)
            and isinstance(canonical, str)
        ):
            continue
        new_name = _opaque_name(src.item_id)
        new_prompt = _rename_def(src.prompt, entry_point, new_name)
        new_canonical = _rename_def(canonical, entry_point, new_name)
        new_test = _rename_def(test_block, entry_point, new_name)
        metadata: dict[str, str | int | float | bool | None] = {
            "canonical_solution": new_canonical,
            "test": new_test,
            "entry_point": new_name,
            "original_entry_point": entry_point,
            "calibration_probe": True,
        }
        new_item = BenchmarkItem(
            item_id=f"calib::{src.item_id}",
            benchmark=f"{src.benchmark}-calibprobe",
            task_family=TaskFamily.CODE,
            prompt=new_prompt,
            reference=new_canonical,
            metadata=metadata,
        )
        out.append(
            ObfuscatedItem(
                item=new_item,
                rewritten_test=new_test,
                rewritten_entry_point=new_name,
                original_entry_point=entry_point,
            )
        )
    return out


def grade_calibration_probe(
    probe: ObfuscatedItem, candidate: str
) -> bool:
    """Run `candidate` against the probe's rewritten test block in the sandbox.

    Returns `True` iff all assertions pass and the script exits cleanly.
    """
    from panoptes.sandbox.python_exec import humaneval_check  # noqa: PLC0415

    result = humaneval_check(
        prompt=probe.item.prompt,
        candidate=candidate,
        test=probe.rewritten_test,
        entry_point=probe.rewritten_entry_point,
    )
    return result.passed
