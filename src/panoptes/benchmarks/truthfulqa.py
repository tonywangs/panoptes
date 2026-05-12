"""TruthfulQA loader (+ optional BM25 evidence retrieval).

TruthfulQA (Lin et al., 2022) generation split, fetched from the HF
datasets parquet mirror. Each item has a `question` and a list of
correct / incorrect references; PANOPTES exposes the question as the
prompt and stores the references for downstream judges.

The BM25-over-Wikipedia evidence retrieval described in the spec is a
bonus path; v1 ships a small `BM25Retriever` over a user-supplied passage
list, with the canonical Wikipedia integration left as a follow-up. The
M5 milestone exists primarily to wire the *generation* benchmark; the
retrieval signal is useful but not load-bearing for v1's acceptance
criteria.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from panoptes.benchmarks.loader import http_fetch_cached
from panoptes.errors import BenchmarkError
from panoptes.schemas import BenchmarkItem, TaskFamily

# Parquet mirror of the generation split.
_TRUTHFULQA_URL = (
    "https://huggingface.co/datasets/truthfulqa/truthful_qa/resolve/main/"
    "generation/validation-00000-of-00001.parquet"
)


def load_truthfulqa(
    *,
    cache_dir: Path,
    limit: int | None = None,
    url: str = _TRUTHFULQA_URL,
) -> list[BenchmarkItem]:
    """Load TruthfulQA generation split from the HF parquet mirror.

    Requires `pyarrow` (installed as a transitive dep of `datasets` in
    the `bench` extra). Returns `BenchmarkItem`s in dataset order.
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:
        raise BenchmarkError(
            "TruthfulQA loader requires pyarrow. Install via: uv sync --extra bench"
        ) from exc
    raw = http_fetch_cached(url, cache_dir=cache_dir, suffix=".parquet")
    table = pq.read_table(io.BytesIO(raw))
    df = table.to_pydict()
    if "question" not in df:
        raise BenchmarkError(
            f"TruthfulQA parquet missing 'question' column; got {sorted(df.keys())}"
        )
    n = len(df["question"])
    items: list[BenchmarkItem] = []
    for i in range(n):
        items.append(_row_to_item(i, df, i))
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise BenchmarkError(f"TruthfulQA payload from {url} was empty")
    return items


def _row_to_item(idx: int, df: dict[str, list[Any]], row: int) -> BenchmarkItem:
    question = df["question"][row]
    if not isinstance(question, str):
        raise BenchmarkError(f"TruthfulQA row {row} has non-str question: {question!r}")
    metadata: dict[str, str | int | float | bool | None] = {}
    for key in ("best_answer", "correct_answers", "incorrect_answers", "category"):
        if key in df:
            value = df[key][row]
            if isinstance(value, list):
                metadata[key] = "\n".join(str(v) for v in value)  # type: ignore[redundant-cast]
            elif value is not None:
                metadata[key] = str(value)
    return BenchmarkItem(
        item_id=f"TruthfulQA/{idx}",
        benchmark="truthfulqa",
        task_family=TaskFamily.FACTUALITY,
        prompt=question,
        reference=str(metadata.get("best_answer", "")) or None,
        metadata=metadata,
    )


class BM25Retriever:
    """Tiny BM25 wrapper for injecting passage evidence into the factuality judge.

    Construct with a list of passages; call `top_k(query, k)` to retrieve
    the most relevant passages. Uses `rank-bm25` if available; falls back
    to a deterministic-by-length stub otherwise (still ordered, but no
    actual lexical scoring).
    """

    def __init__(self, passages: Sequence[str]) -> None:
        self._passages = list(passages)
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415

            tokenized = [p.lower().split() for p in self._passages]
            self._bm25: Any | None = BM25Okapi(tokenized)
        except ImportError:
            self._bm25 = None

    def top_k(self, query: str, k: int = 5) -> list[str]:
        if not self._passages:
            return []
        if self._bm25 is None:
            # Stub fallback: deterministic order by descending length, take top k.
            return sorted(self._passages, key=lambda p: -len(p))[:k]
        scores = self._bm25.get_scores(query.lower().split())
        order = sorted(
            range(len(self._passages)), key=lambda i: -float(scores[i])
        )
        return [self._passages[i] for i in order[:k]]
