"""End-to-end pipeline smoke test with the deterministic mock client.

Runs the full M1 pipeline (judge + storage + conformal) against two synthetic
benchmark items, asserts duckdb gets two rows, and that the cost report's
shape is consistent with what the mock judge produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from panoptes.clients._mock import MockClient
from panoptes.judges.base import PromptTemplate
from panoptes.judges.rubric import RubricJudge
from panoptes.pipeline import EvalConfig, JudgeRef, new_run_id, run_evaluation
from panoptes.schemas import BenchmarkItem, TaskFamily
from panoptes.storage.duckdb_store import DuckDBStore
from panoptes.storage.prompt_cache import PromptCache


def _make_template() -> PromptTemplate:
    return PromptTemplate(
        system="You are a test judge. Use record_score.",
        user="[Task]\n{prompt}\n\n[Response]\n{response}",
        content_hash="testhash00000001",
        source_path=Path("/dev/null"),
    )


@pytest.mark.asyncio
async def test_pipeline_writes_rows_and_aggregates(tmp_path: Path) -> None:
    items = [
        BenchmarkItem(
            item_id=f"item/{i}",
            benchmark="testbench",
            task_family=TaskFamily.CODE,
            prompt=f"prompt {i}",
        )
        for i in range(2)
    ]
    responses = {item.item_id: f"response to {item.item_id}" for item in items}

    client = MockClient(provider="anthropic", model="claude-sonnet-4-6")
    judge = RubricJudge(client=client, template=_make_template(), variant="rubric_code_v1")
    judge_ref = JudgeRef(judge=judge, prompt_version_hash="testhash00000001")

    db_path = tmp_path / "smoke.duckdb"
    with DuckDBStore.open(db_path) as store:
        cache = PromptCache(store=store)
        cost = await run_evaluation(
            items=items,
            responses=responses,
            judges=[judge_ref],
            store=store,
            cache=cache,
            config=EvalConfig(run_id=new_run_id(), alpha=0.1, uq_methods=("split",)),
            model_under_test="test_model",
        )
        assert store.count_rows() == 2
        assert cost.n_calls == 2
        assert cost.usd_total >= 0.0
        # Each call charges the mocked usage (200 in, 60 out at sonnet pricing).
        expected_usd = 2 * (200 * 3.0 + 60 * 15.0) / 1_000_000.0
        assert cost.usd_total == pytest.approx(expected_usd, abs=1e-9)


@pytest.mark.asyncio
async def test_pipeline_respects_prompt_cache(tmp_path: Path) -> None:
    """Second run with the same template_hash should skip cached items."""
    items = [
        BenchmarkItem(
            item_id="item/A",
            benchmark="testbench",
            task_family=TaskFamily.CODE,
            prompt="p",
        )
    ]
    responses = {"item/A": "r"}
    client = MockClient(provider="anthropic", model="claude-sonnet-4-6")
    judge = RubricJudge(client=client, template=_make_template(), variant="rubric_code_v1")
    ref = JudgeRef(judge=judge, prompt_version_hash="testhash00000001")

    db_path = tmp_path / "cache_test.duckdb"
    with DuckDBStore.open(db_path) as store:
        cache = PromptCache(store=store)
        first = await run_evaluation(
            items=items,
            responses=responses,
            judges=[ref],
            store=store,
            cache=cache,
            config=EvalConfig(run_id=new_run_id()),
            model_under_test="m",
        )
        second = await run_evaluation(
            items=items,
            responses=responses,
            judges=[ref],
            store=store,
            cache=cache,
            config=EvalConfig(run_id=new_run_id()),
            model_under_test="m",
        )
        assert first.n_calls == 1
        assert second.n_calls == 0  # cache hit
        assert store.count_rows() == 1
