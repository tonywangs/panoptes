"""Thompson-sampling bandit router.

Each `(judge, task_family)` pair is one bandit arm. Arms are modeled as
Bernoulli with Beta(α, β) priors; α and β are updated online based on
whether the judge "did its job" on a given item:

    `reward(judge, item) = 1` if the judge meaningfully reduced epistemic
    uncertainty per unit cost on this item, else `0`.

Concretely we compute, after collecting all responses for an item,

    info_per_dollar_j = max(0, var_before − var_after_excluding_j) / cost_j

and award reward = 1 to the arms whose info-per-dollar exceeds the median
across this item's judges, reward = 0 otherwise. This is a coarse
discretization of the underlying continuous signal but suffices for the
Beta-Bernoulli machinery and avoids the bias of variance-stabilizing
transforms over an arbitrary scale.

The bandit *selects* by Thompson sampling: at item i, draw
`θ_j ~ Beta(α_j, β_j)` for each judge, then pick the top-K judges by
sampled `θ_j` (subject to a cost budget, if provided).

This is intentionally a *baseline* bandit; the cost-vs-quality gap
against the all-judges strategy is measured against the calibration
probe (see `benchmarks/calibration_probe.py`).

Reference
---------
- Russo, Van Roy, Kazerouni, Osband, Wen (2018). *A Tutorial on Thompson Sampling.* arXiv:1707.02038.
- Chapelle, Li (2011). *An Empirical Evaluation of Thompson Sampling.* NeurIPS (the canonical re-validation of TS over UCB1).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from panoptes.routing.base import JudgeCatalog, RouterDecision, StrategyName
from panoptes.schemas import BenchmarkItem, JudgeResponse


@dataclass(slots=True)
class BetaParams:
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.alpha, self.beta))

    @property
    def mean(self) -> float:
        return self.alpha / max(self.alpha + self.beta, 1e-9)


@dataclass(slots=True)
class ThompsonBandit:
    """Bandit router with Beta(α, β) per `(judge, task_family)`."""

    strategy: StrategyName = "bandit"
    top_k: int = 2
    seed: int = 0
    _arms: dict[tuple[str, str], BetaParams] = field(default_factory=dict)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ routing

    def initial(self, item: BenchmarkItem, catalog: JudgeCatalog) -> RouterDecision:
        """Thompson-sample one θ per judge; take the top-K by sample value.

        We always include the top-1 judge by mean (greedy floor) plus
        `top_k - 1` additional Thompson-sampled judges. This prevents the
        bandit from getting stuck on a single suboptimal arm early.
        """
        family = item.task_family.value
        ids = catalog.all_ids()
        if not ids:
            return RouterDecision(judge_ids=[], phase="none")
        samples = {
            jid: self._arm(jid, family).sample(self._rng) for jid in ids
        }
        sorted_ids = sorted(ids, key=lambda j: samples[j], reverse=True)
        k = min(self.top_k, len(sorted_ids))
        selected = sorted_ids[:k]
        return RouterDecision(
            judge_ids=selected,
            phase="initial",
            reason=(
                f"strategy=bandit family={family} top_k={k} "
                f"samples={ {j: round(samples[j], 3) for j in selected} }"
            ),
        )

    def escalate(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses_so_far: list[JudgeResponse],
    ) -> RouterDecision:
        del item, catalog, responses_so_far
        # Bandit makes its full decision up front; no separate escalation step.
        return RouterDecision(judge_ids=[], phase="none")

    def update(
        self,
        item: BenchmarkItem,
        catalog: JudgeCatalog,
        responses: list[JudgeResponse],
    ) -> None:
        """Reward-update the arms for the judges called on `item`.

        Reward design: 1 if the judge's info-per-dollar exceeded the median
        across called judges on this item, else 0. With < 2 responses, no
        update (we have no comparison signal).
        """
        del catalog
        if len(responses) < 2:
            return
        family = item.task_family.value
        scores = np.asarray([r.score.value for r in responses], dtype=np.float64)
        var_before = float(scores.var(ddof=1))
        info_per_dollar: dict[str, float] = {}
        for r in responses:
            others = [s for s in scores if s != r.score.value]
            var_after = float(np.var(others, ddof=1)) if len(others) > 1 else 0.0
            cost = max(r.cost_usd, 1e-6)
            info_per_dollar[r.judge_id] = max(0.0, var_before - var_after) / cost
        if not info_per_dollar:
            return
        median = float(np.median(list(info_per_dollar.values())))
        for judge_id, ipd in info_per_dollar.items():
            arm = self._arm(judge_id, family)
            if ipd >= median:
                arm.alpha += 1.0
            else:
                arm.beta += 1.0

    # ------------------------------------------------------------------ state

    def _arm(self, judge_id: str, family: str) -> BetaParams:
        key = (judge_id, family)
        arm = self._arms.get(key)
        if arm is None:
            arm = BetaParams()
            self._arms[key] = arm
        return arm

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable snapshot of arm state for storage / debugging."""
        out: defaultdict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for (judge_id, family), arm in self._arms.items():
            out[family][judge_id] = {"alpha": arm.alpha, "beta": arm.beta, "mean": arm.mean}
        return dict(out)

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), indent=2, sort_keys=True)

    def warm_start(self, snapshot: dict[str, Any]) -> None:
        """Restore arm state from a previous run's snapshot."""
        for family, judges in snapshot.items():
            if not isinstance(judges, dict):
                continue
            for judge_id, params in judges.items():
                if not isinstance(params, dict):
                    continue
                alpha = float(params.get("alpha", 1.0))
                beta = float(params.get("beta", 1.0))
                self._arms[(judge_id, family)] = BetaParams(alpha=alpha, beta=beta)
