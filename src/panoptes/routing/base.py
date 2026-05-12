"""`JuryRouter` Protocol + supporting value types.

A router decides, per item, which subset of judges to call. The pipeline
calls `decide(item, history)` once before the point pass and again *after*
the point pass if the strategy supports adaptive escalation (e.g.
`EscalationPolicy` will return additional judges if posterior variance
exceeds its threshold).

`JudgeMeta` carries enough metadata for routing decisions (cost tier,
expected latency) without coupling the router to the concrete `Judge`
implementation. Strategies that don't need these fields (`AllJudges`,
`SingleJudge`) ignore them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from panoptes.schemas import BenchmarkItem, JudgeResponse

StrategyName = Literal["all", "single", "escalation", "bandit"]


@dataclass(frozen=True, slots=True)
class JudgeMeta:
    """Per-judge metadata visible to the router.

    `cost_tier` is a coarse-grained classifier (cheap / mid / expensive)
    rather than a USD/token estimate so the router can rank judges without
    needing per-item cost forecasts. `expected_latency_ms` is informational
    for escalation budget calculations; defaults are best-effort and may
    be overridden by callers.
    """

    judge_id: str
    cost_tier: Literal["cheap", "mid", "expensive"] = "mid"
    expected_latency_ms: float = 1000.0


@dataclass(slots=True)
class JudgeCatalog:
    """The pool of judges available for routing.

    The catalog is the single source of truth for "what judges exist in
    this run"; routers index into it by `judge_id`.
    """

    judges: list[JudgeMeta]

    def by_id(self) -> Mapping[str, JudgeMeta]:
        return {j.judge_id: j for j in self.judges}

    def all_ids(self) -> list[str]:
        return [j.judge_id for j in self.judges]

    def by_cost_tier(
        self, tier: Literal["cheap", "mid", "expensive"]
    ) -> list[JudgeMeta]:
        return [j for j in self.judges if j.cost_tier == tier]


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """One routing decision: the judges to call, plus a free-text reason.

    `phase` distinguishes the initial selection from any escalation
    add-ons; the pipeline uses it to log the decision rationale alongside
    the item.
    """

    judge_ids: list[str]
    phase: Literal["initial", "escalation", "none"] = "initial"
    reason: str = ""


@runtime_checkable
class JuryRouter(Protocol):
    """Strategy that picks which judges to call for an item.

    The Protocol is intentionally non-async; routing decisions are
    cheap (no network) and live inside the pipeline's async loop.
    """

    strategy: StrategyName

    def initial(self, item: BenchmarkItem, catalog: JudgeCatalog) -> RouterDecision:
        """Pre-point-pass selection."""
        ...

    def escalate(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses_so_far: list[JudgeResponse],
    ) -> RouterDecision:
        """Post-point-pass: return any additional judges to call.

        Strategies that don't escalate return `RouterDecision([], phase="none")`.
        """
        ...

    def update(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses: list[JudgeResponse],
    ) -> None:
        """Online update — called after every item is finished.

        Stateless strategies are no-ops; the bandit uses this to update
        its Beta posteriors. Default protocol implementations are not
        required to mutate; the contract is "may mutate self".
        """
        ...
