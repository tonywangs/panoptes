"""MBPP loader.

MBPP (Austin et al., 2021) is a 974-problem Python programming benchmark
shipped as a single JSONL. Each problem has a natural-language `text`
prompt, a `code` reference solution, and a `test_list` of assert
statements that gate correctness.

We treat each problem as `task_family=CODE`, like HumanEval, with the
canonical solution and test list stored in `BenchmarkItem.metadata`.
The official upstream URL is the source-of-truth artifact in the
google-research repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from panoptes.benchmarks.loader import http_fetch_cached, iter_jsonl
from panoptes.errors import BenchmarkError
from panoptes.schemas import BenchmarkItem, TaskFamily

_MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
)


def load_mbpp(
    *,
    cache_dir: Path,
    limit: int | None = None,
    url: str = _MBPP_URL,
) -> list[BenchmarkItem]:
    """Fetch + parse MBPP into `BenchmarkItem`s ordered as upstream emits them."""
    raw = http_fetch_cached(url, cache_dir=cache_dir, suffix=".jsonl")
    items: list[BenchmarkItem] = []
    for record in iter_jsonl(raw):
        items.append(_to_item(record))
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise BenchmarkError(f"MBPP payload from {url} was empty")
    return items


def _to_item(record: dict[str, Any]) -> BenchmarkItem:
    task_id = record.get("task_id")
    text = record.get("text")
    code = record.get("code")
    tests = record.get("test_list")
    if not isinstance(text, str) or task_id is None:
        raise BenchmarkError(f"MBPP record missing text/task_id: {record!r}")
    metadata: dict[str, str | int | float | bool | None] = {}
    if isinstance(code, str):
        metadata["canonical_solution"] = code
    if isinstance(tests, list):
        metadata["test_list_json"] = "\n".join(str(t) for t in tests)  # type: ignore[redundant-cast]
    return BenchmarkItem(
        item_id=f"MBPP/{task_id}",
        benchmark="mbpp",
        task_family=TaskFamily.CODE,
        prompt=text,
        reference=code if isinstance(code, str) else None,
        metadata=metadata,
    )
