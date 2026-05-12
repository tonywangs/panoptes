"""Routing strategies: decide which judges to call per item.

Strategies share the `JuryRouter` Protocol in `base.py`. M3 ships:

- `AllJudges` — call every judge (baseline)
- `SingleJudge` — call exactly one judge (cheapest baseline)
- `EscalationPolicy` — cheap-first; escalate when epistemic uncertainty
  exceeds a threshold
- `ThompsonBandit` — Beta-Bernoulli Thompson sampling over (judge,
  task_family) arms with reward = information per dollar

The pipeline asks the router which judges to call *before* doing the
point pass for each item. Stateful routers (escalation, bandit) update
internally after seeing each item's responses.
"""

from panoptes.routing.bandit import ThompsonBandit
from panoptes.routing.base import JudgeCatalog, JudgeMeta, JuryRouter, RouterDecision
from panoptes.routing.strategies import AllJudges, EscalationPolicy, SingleJudge

__all__ = [
    "AllJudges",
    "EscalationPolicy",
    "JudgeCatalog",
    "JudgeMeta",
    "JuryRouter",
    "RouterDecision",
    "SingleJudge",
    "ThompsonBandit",
]
