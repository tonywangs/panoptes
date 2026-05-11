"""Round-trip and validation tests for the public Pydantic schemas."""

from __future__ import annotations

import pytest

from panoptes.schemas import (
    BenchmarkItem,
    ConformalResult,
    CostReport,
    JudgeResponse,
    RubricScore,
    ScoreScale,
    TaskFamily,
    TokenUsage,
)


def test_benchmark_item_round_trip() -> None:
    item = BenchmarkItem(
        item_id="HumanEval/0",
        benchmark="humaneval",
        task_family=TaskFamily.CODE,
        prompt="def foo():\n    pass",
        reference="    return 42",
        metadata={"entry_point": "foo"},
    )
    blob = item.model_dump_json()
    restored = BenchmarkItem.model_validate_json(blob)
    assert restored == item
    assert restored.task_family is TaskFamily.CODE


def test_rubric_score_clamps_value() -> None:
    with pytest.raises(ValueError, match="less than or equal"):
        RubricScore(value=1.5, rationale="bad")
    with pytest.raises(ValueError, match="greater than or equal"):
        RubricScore(value=-0.1, rationale="bad")


def test_judge_response_round_trip() -> None:
    response = JudgeResponse(
        judge_id="anthropic:claude-sonnet-4-6:rubric_code_v1",
        item_id="HumanEval/0",
        score=RubricScore(value=0.8, scale=ScoreScale.CONTINUOUS, rationale="ok"),
        raw_text="",
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        cost_usd=0.0015,
        latency_ms=350.0,
        prompt_hash="abc1234567890def",
        sampled_at_temperature=0.0,
    )
    blob = response.model_dump_json()
    restored = JudgeResponse.model_validate_json(blob)
    assert restored.judge_id == response.judge_id
    assert restored.score == response.score
    assert restored.usage == response.usage


def test_conformal_result_validates_bounds() -> None:
    cr = ConformalResult(method="split", alpha=0.1, point=0.5, lo=0.3, hi=0.7)
    assert cr.lo <= cr.point <= cr.hi
    with pytest.raises(ValueError, match="less than 1"):
        ConformalResult(method="split", alpha=1.0, point=0.5, lo=0.3, hi=0.7)


def test_cost_report_defaults_to_zero() -> None:
    cr = CostReport(usd_total=0.0)
    assert cr.usd_total == 0.0
    assert cr.input_tokens == 0
    assert cr.by_judge == {}
