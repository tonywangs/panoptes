"""Stateless / lightly-stateful routing strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panoptes.routing.base import (
    JudgeCatalog,
    RouterDecision,
    StrategyName,
)
from panoptes.schemas import BenchmarkItem, JudgeResponse


@dataclass(slots=True)
class AllJudges:
    """Always call every judge. The baseline; never escalates."""

    strategy: StrategyName = "all"

    def initial(self, item: BenchmarkItem, catalog: JudgeCatalog) -> RouterDecision:
        del item
        return RouterDecision(
            judge_ids=catalog.all_ids(), phase="initial", reason="strategy=all"
        )

    def escalate(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses_so_far: list[JudgeResponse],
    ) -> RouterDecision:
        del item, catalog, responses_so_far
        return RouterDecision(judge_ids=[], phase="none")

    def update(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses: list[JudgeResponse],
    ) -> None:
        del item, catalog, responses


@dataclass(slots=True)
class SingleJudge:
    """Always call exactly one judge (the cheapest by default, or a named one)."""

    strategy: StrategyName = "single"
    judge_id: str | None = None  # None means "pick the cheapest"

    def initial(self, item: BenchmarkItem, catalog: JudgeCatalog) -> RouterDecision:
        del item
        if self.judge_id is not None:
            if self.judge_id not in catalog.by_id():
                raise ValueError(
                    f"SingleJudge configured for {self.judge_id!r} but it is not "
                    f"in the catalog: {catalog.all_ids()}"
                )
            return RouterDecision(
                judge_ids=[self.judge_id],
                phase="initial",
                reason=f"strategy=single judge={self.judge_id}",
            )
        # Otherwise: pick the cheapest tier, breaking ties by judge_id order.
        for tier in ("cheap", "mid", "expensive"):
            picks = catalog.by_cost_tier(tier)  # type: ignore[arg-type]
            if picks:
                chosen = sorted(picks, key=lambda j: j.judge_id)[0]
                return RouterDecision(
                    judge_ids=[chosen.judge_id],
                    phase="initial",
                    reason=f"strategy=single tier={tier} judge={chosen.judge_id}",
                )
        return RouterDecision(judge_ids=[], phase="none", reason="empty catalog")

    def escalate(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses_so_far: list[JudgeResponse],
    ) -> RouterDecision:
        del item, catalog, responses_so_far
        return RouterDecision(judge_ids=[], phase="none")

    def update(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses: list[JudgeResponse],
    ) -> None:
        del item, catalog, responses


@dataclass(slots=True)
class EscalationPolicy:
    """Cheap-first, escalate to a stronger judge when epistemic variance is high.

    Behavior:
        1. Initial pass: call all `cheap`-tier judges (or `mid` if no `cheap`).
        2. After the point pass, compute inter-judge variance on the scores.
           If variance > `tau`, call one `expensive`-tier judge (the first one
           we haven't already called).

    This is a deterministic stand-in for the bandit; it generalizes to
    "call the next-most-informative judge until we're confident."
    """

    strategy: StrategyName = "escalation"
    tau: float = 0.02  # variance threshold; tune to your score scale
    cheap_tier: str = "cheap"

    def initial(self, item: BenchmarkItem, catalog: JudgeCatalog) -> RouterDecision:
        del item
        cheap = [j.judge_id for j in catalog.by_cost_tier(self.cheap_tier)]  # type: ignore[arg-type]
        if not cheap:
            cheap = [j.judge_id for j in catalog.by_cost_tier("mid")]
        if not cheap:
            cheap = catalog.all_ids()
        return RouterDecision(
            judge_ids=cheap, phase="initial", reason=f"strategy=escalation cheap_tier={self.cheap_tier}"
        )

    def escalate(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses_so_far: list[JudgeResponse],
    ) -> RouterDecision:
        del item
        if len(responses_so_far) < 2:
            return RouterDecision(judge_ids=[], phase="none")
        values = np.asarray([r.score.value for r in responses_so_far], dtype=np.float64)
        variance = float(values.var(ddof=1))
        if variance <= self.tau:
            return RouterDecision(
                judge_ids=[],
                phase="none",
                reason=f"variance={variance:.4f} ≤ tau={self.tau}",
            )
        already_called = {r.judge_id for r in responses_so_far}
        expensive = [
            j.judge_id
            for j in catalog.by_cost_tier("expensive")
            if j.judge_id not in already_called
        ]
        if not expensive:
            return RouterDecision(judge_ids=[], phase="none", reason="no expensive judge available")
        return RouterDecision(
            judge_ids=[expensive[0]],
            phase="escalation",
            reason=(
                f"variance={variance:.4f} > tau={self.tau}; escalating to {expensive[0]}"
            ),
        )

    def update(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses: list[JudgeResponse],
    ) -> None:
        del item, catalog, responses
