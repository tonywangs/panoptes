"""Prompt-template content cache, layered over `DuckDBStore`.

The contract is simple: given a `(benchmark, judge_id, item_id,
prompt_version_hash)` key, the cache tells the pipeline whether it has
already evaluated this combination at temperature=0 (the point-estimate
pass). If yes, the pipeline skips the call; if no, it issues the request,
writes the resulting `JudgeRow`, and future runs hit the cache.

Sampling-pass rows (`sample_index > 0`) are deliberately excluded from
this cache: they are MC samples used by sampling-based UQ, and re-sampling
on re-run is the *correct* behavior since each sample is an independent
draw from the judge's response distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

from panoptes.storage.duckdb_store import DuckDBStore


@dataclass(slots=True)
class PromptCache:
    """Thin wrapper around `DuckDBStore` exposing the cache contract.

    Kept as a separate type so the pipeline depends on the *contract*
    (a `PromptCache`) rather than the storage backend, making it easy to
    swap in an in-memory cache for tests.
    """

    store: DuckDBStore

    def is_cached(
        self,
        *,
        benchmark: str,
        judge_id: str,
        item_id: str,
        prompt_version_hash: str,
    ) -> bool:
        return self.store.has_cached(
            benchmark=benchmark,
            judge_id=judge_id,
            item_id=item_id,
            prompt_version_hash=prompt_version_hash,
        )
