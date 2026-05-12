"""GSM8K loader.

GSM8K (Cobbe et al. 2021) is a grade-school math word-problem benchmark
with explicit chain-of-thought reference answers. The reference answer is
embedded in the `answer` field as `... #### 42`; we parse out the final
number for use as a verifiable ground truth.

PANOPTES uses this benchmark to *compare* a verifiable signal against
LLM-judge graded scores — the calibration probe for the "is the judge
noisy" question.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from panoptes.benchmarks.loader import http_fetch_cached, iter_jsonl
from panoptes.errors import BenchmarkError
from panoptes.schemas import BenchmarkItem, TaskFamily

# Test-split JSONL from the openai/grade-school-math GitHub repo.
_GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)

_FINAL_NUMBER_RE = re.compile(r"####\s*(?P<value>-?[\d,]+(?:\.\d+)?)")


def parse_final_answer(answer: str) -> str | None:
    """Extract the canonical `#### <num>` final-answer marker from GSM8K text."""
    match = _FINAL_NUMBER_RE.search(answer)
    if match is None:
        return None
    return match.group("value").replace(",", "").strip()


def load_gsm8k(
    *,
    cache_dir: Path,
    limit: int | None = None,
    url: str = _GSM8K_URL,
) -> list[BenchmarkItem]:
    """Fetch and parse GSM8K test split into `BenchmarkItem`s."""
    raw = http_fetch_cached(url, cache_dir=cache_dir, suffix=".jsonl")
    items: list[BenchmarkItem] = []
    for i, record in enumerate(iter_jsonl(raw)):
        items.append(_to_item(i, record))
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise BenchmarkError(f"GSM8K payload from {url} was empty")
    return items


def _to_item(idx: int, record: dict[str, Any]) -> BenchmarkItem:
    question = record.get("question")
    answer = record.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise BenchmarkError(f"GSM8K record missing question/answer: {record!r}")
    final = parse_final_answer(answer)
    metadata: dict[str, str | int | float | bool | None] = {
        "reference_chain_of_thought": answer,
    }
    if final is not None:
        metadata["final_answer"] = final
    return BenchmarkItem(
        item_id=f"GSM8K/{idx}",
        benchmark="gsm8k",
        task_family=TaskFamily.MATH,
        prompt=question,
        reference=final,
        metadata=metadata,
    )
