"""MT-Bench loader (lmsys/FastChat distribution).

MT-Bench (Zheng et al., 2023) is 80 prompts × 2 turns of open-ended chat,
plus an LLM-judge rubric. PANOPTES treats each *first turn* as a single
`BenchmarkItem` with `task_family=FREEFORM`; the optional second turn is
stored in `metadata['turn_2']`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from panoptes.benchmarks.loader import http_fetch_cached, iter_jsonl
from panoptes.errors import BenchmarkError
from panoptes.schemas import BenchmarkItem, TaskFamily

_MTBENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


def load_mtbench(
    *,
    cache_dir: Path,
    limit: int | None = None,
    url: str = _MTBENCH_URL,
) -> list[BenchmarkItem]:
    """Fetch the MT-Bench question set as `BenchmarkItem`s.

    Only the first turn is used as the `prompt`; subsequent turns are
    available under `metadata['turn_N']` for callers that want them.
    """
    raw = http_fetch_cached(url, cache_dir=cache_dir, suffix=".jsonl")
    items: list[BenchmarkItem] = []
    for record in iter_jsonl(raw):
        items.append(_to_item(record))
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise BenchmarkError(f"MT-Bench payload from {url} was empty")
    return items


def _to_item(record: dict[str, Any]) -> BenchmarkItem:
    qid = record.get("question_id")
    turns = record.get("turns")
    category = record.get("category", "")
    if qid is None or not isinstance(turns, list) or not turns:
        raise BenchmarkError(f"MT-Bench record missing question_id/turns: {record!r}")
    metadata: dict[str, str | int | float | bool | None] = {
        "category": str(category),
    }
    for i, turn in enumerate(turns[1:], start=2):
        if isinstance(turn, str):
            metadata[f"turn_{i}"] = turn
    return BenchmarkItem(
        item_id=f"MTBench/{qid}",
        benchmark="mtbench",
        task_family=TaskFamily.FREEFORM,
        prompt=str(turns[0]),
        reference=None,
        metadata=metadata,
    )
