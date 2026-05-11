"""HumanEval loader.

HumanEval (Chen et al., 2021, *Evaluating Large Language Models Trained on
Code*) is a 164-problem Python coding benchmark distributed by OpenAI as a
gzipped JSONL. Each problem has a function signature + docstring (`prompt`),
a reference (`canonical_solution`), and a unit-test block (`test`).

For M1 we treat each problem as a `BenchmarkItem` with `task_family=CODE`.
The reference solution lives in `metadata['canonical_solution']`; the test
block lives in `metadata['test']` for downstream sandboxed execution (M5).
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from panoptes.benchmarks.loader import http_fetch_cached, iter_jsonl
from panoptes.errors import BenchmarkError
from panoptes.schemas import BenchmarkItem, TaskFamily

# Canonical mirror. The file is small (~200 kB gzipped) and stable; we cache
# locally on first fetch. If GitHub raw fails, set PANOPTES_HUMANEVAL_URL.
_HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
)


def load_humaneval(
    *,
    cache_dir: Path,
    limit: int | None = None,
    url: str = _HUMANEVAL_URL,
) -> list[BenchmarkItem]:
    """Fetch and parse HumanEval into a list of `BenchmarkItem`s.

    `limit` truncates the dataset for smoke / dev runs (`--n 5` in the CLI).
    Items are returned in the order they appear in the upstream JSONL,
    which is the canonical ordering used in HumanEval pass@k papers.
    """
    raw_gz = http_fetch_cached(url, cache_dir=cache_dir, suffix=".jsonl.gz")
    try:
        raw = gzip.decompress(raw_gz)
    except OSError as exc:
        raise BenchmarkError(f"Failed to decompress HumanEval payload from {url}") from exc
    items: list[BenchmarkItem] = []
    for record in iter_jsonl(raw):
        items.append(_to_item(record))
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise BenchmarkError(f"HumanEval payload from {url} was empty")
    return items


def _to_item(record: dict[str, Any]) -> BenchmarkItem:
    task_id_raw = record.get("task_id")
    prompt = record.get("prompt")
    canonical = record.get("canonical_solution")
    test = record.get("test")
    entry_point = record.get("entry_point")
    if not isinstance(task_id_raw, str) or not isinstance(prompt, str):
        raise BenchmarkError(
            f"HumanEval record missing required str fields task_id/prompt: {record!r}"
        )
    metadata: dict[str, str | int | float | bool | None] = {}
    if isinstance(canonical, str):
        metadata["canonical_solution"] = canonical
    if isinstance(test, str):
        metadata["test"] = test
    if isinstance(entry_point, str):
        metadata["entry_point"] = entry_point
    return BenchmarkItem(
        item_id=task_id_raw,
        benchmark="humaneval",
        task_family=TaskFamily.CODE,
        prompt=prompt,
        reference=canonical if isinstance(canonical, str) else None,
        metadata=metadata,
    )


def humaneval_iter(items: Iterable[BenchmarkItem]) -> Iterable[BenchmarkItem]:
    """Identity passthrough that documents iteration intent for the pipeline."""
    yield from items
