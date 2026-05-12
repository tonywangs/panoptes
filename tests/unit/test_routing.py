"""Tests for routing strategies."""

from __future__ import annotations

from panoptes.routing.bandit import ThompsonBandit
from panoptes.routing.base import JudgeCatalog, JudgeMeta
from panoptes.routing.strategies import AllJudges, EscalationPolicy, SingleJudge
from panoptes.schemas import (
    BenchmarkItem,
    JudgeResponse,
    RubricScore,
    ScoreScale,
    TaskFamily,
    TokenUsage,
)


def _item(family: TaskFamily = TaskFamily.CODE) -> BenchmarkItem:
    return BenchmarkItem(
        item_id="i", benchmark="test", task_family=family, prompt="p"
    )


def _catalog() -> JudgeCatalog:
    return JudgeCatalog(
        judges=[
            JudgeMeta(judge_id="cheap-A", cost_tier="cheap"),
            JudgeMeta(judge_id="cheap-B", cost_tier="cheap"),
            JudgeMeta(judge_id="mid-A", cost_tier="mid"),
            JudgeMeta(judge_id="expensive-A", cost_tier="expensive"),
        ]
    )


def _response(judge_id: str, score: float, cost: float = 0.001) -> JudgeResponse:
    return JudgeResponse(
        judge_id=judge_id,
        item_id="i",
        score=RubricScore(value=score, scale=ScoreScale.CONTINUOUS, rationale=""),
        raw_text="",
        usage=TokenUsage(input_tokens=10, output_tokens=10),
        cost_usd=cost,
        latency_ms=100.0,
        prompt_hash="x",
        sampled_at_temperature=0.0,
    )


def test_all_judges_returns_full_catalog() -> None:
    router = AllJudges()
    decision = router.initial(_item(), _catalog())
    assert set(decision.judge_ids) == {"cheap-A", "cheap-B", "mid-A", "expensive-A"}
    # No escalation regardless of input.
    esc = router.escalate(_item(), _catalog(), [_response("cheap-A", 0.5)])
    assert esc.judge_ids == []


def test_single_judge_picks_cheapest_by_default() -> None:
    router = SingleJudge()
    decision = router.initial(_item(), _catalog())
    assert decision.judge_ids == ["cheap-A"]


def test_single_judge_honors_explicit_id() -> None:
    router = SingleJudge(judge_id="mid-A")
    decision = router.initial(_item(), _catalog())
    assert decision.judge_ids == ["mid-A"]


def test_escalation_fires_when_variance_exceeds_tau() -> None:
    router = EscalationPolicy(tau=0.01)
    catalog = _catalog()
    # Two cheap responses with high inter-judge variance.
    responses = [_response("cheap-A", 0.2), _response("cheap-B", 0.8)]
    decision = router.escalate(_item(), catalog, responses)
    assert decision.judge_ids == ["expensive-A"]


def test_escalation_skips_when_variance_below_tau() -> None:
    router = EscalationPolicy(tau=0.1)
    catalog = _catalog()
    # Two cheap responses with low inter-judge variance.
    responses = [_response("cheap-A", 0.50), _response("cheap-B", 0.55)]
    decision = router.escalate(_item(), catalog, responses)
    assert decision.judge_ids == []


def test_bandit_initial_returns_top_k_judges() -> None:
    router = ThompsonBandit(top_k=2, seed=42)
    decision = router.initial(_item(), _catalog())
    assert len(decision.judge_ids) == 2


def test_bandit_converges_to_winning_arm_on_consistent_reward() -> None:
    """Repeatedly reward the same judge → bandit should pick it more often.

    We simulate updates where 'winner' consistently lands above the median
    info-per-dollar, and check that after enough rounds its sampled θ is
    typically higher than competitors'.
    """
    catalog = _catalog()
    router = ThompsonBandit(top_k=2, seed=0)
    item = _item()
    for _ in range(50):
        # Synthesize responses where 'winner' has spread that drops variance.
        responses = [
            _response("cheap-A", 0.5, cost=0.001),   # winner: drives down variance
            _response("cheap-B", 0.9, cost=0.001),
            _response("mid-A", 0.5, cost=0.005),     # tied with winner but more $
        ]
        router.update(item, catalog, responses)

    # After convergence, the winning arm should have higher posterior mean
    # than the non-winners.
    snap = router.snapshot()
    family = item.task_family.value
    means = {jid: snap[family][jid]["mean"] for jid in snap[family]}
    # cheap-A's mean (winner) should beat at least one consistently-loser arm.
    assert means["cheap-A"] >= means.get("cheap-B", 0.0)
