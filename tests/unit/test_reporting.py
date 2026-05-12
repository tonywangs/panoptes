"""Smoke test for `panoptes.reporting.render_report`."""

from __future__ import annotations

from pathlib import Path

import pytest

from panoptes.clients._mock import MockClient
from panoptes.judges.base import PromptTemplate
from panoptes.judges.rubric import RubricJudge
from panoptes.pipeline import EvalConfig, JudgeRef, new_run_id, run_evaluation
from panoptes.reporting import render_report, write_report
from panoptes.schemas import BenchmarkItem, TaskFamily
from panoptes.storage.duckdb_store import DuckDBStore


def _make_template() -> PromptTemplate:
    return PromptTemplate(
        system="test", user="[Task]\n{prompt}\n\n[Response]\n{response}",
        content_hash="x", source_path=Path("/dev/null"),
    )


@pytest.mark.asyncio
async def test_report_renders_with_runs_and_judges(tmp_path: Path) -> None:
    items = [
        BenchmarkItem(
            item_id=f"i/{k}", benchmark="t", task_family=TaskFamily.CODE, prompt=f"p{k}"
        )
        for k in range(5)
    ]
    responses = {it.item_id: f"r{it.item_id}" for it in items}
    judges = [
        JudgeRef(
            judge=RubricJudge(
                client=MockClient(provider="anthropic", model="claude-haiku-4-5"),
                template=_make_template(),
                variant="v",
            ),
            prompt_version_hash="x",
            cost_tier="cheap",
        ),
        JudgeRef(
            judge=RubricJudge(
                client=MockClient(provider="anthropic", model="claude-sonnet-4-6"),
                template=_make_template(),
                variant="v",
            ),
            prompt_version_hash="x",
            cost_tier="mid",
        ),
    ]
    db = tmp_path / "report.duckdb"
    with DuckDBStore.open(db) as store:
        await run_evaluation(
            items=items, responses=responses, judges=judges, store=store,
            cache=None, config=EvalConfig(run_id=new_run_id()), model_under_test="m",
        )
    html_text = render_report(db)
    assert "PANOPTES report" in html_text
    assert "Cost by judge" in html_text
    assert "Inter-judge agreement" in html_text

    out_path = tmp_path / "report.html"
    write_report(db, out_path)
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("<!doctype html>")
