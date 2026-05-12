"""Property-based invariants for UQ methods.

Asserts behavior that must hold regardless of input:

- `MondrianConformal` per-group quantile monotone in `1 - alpha`.
- `semantic_entropy` always in `[0, log N]`.
- `self_consistency_stats` CI bracket is non-negative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from panoptes.uq.conformal_mondrian import MondrianConformal
from panoptes.uq.nli.base import NLILabel, NLIScores
from panoptes.uq.self_consistency import self_consistency_stats
from panoptes.uq.semantic_entropy import max_entropy, semantic_entropy

_PRED_LABEL_STRAT = st.lists(
    st.tuples(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
    min_size=10,
    max_size=200,
)


@settings(max_examples=40, deadline=None)
@given(
    _PRED_LABEL_STRAT,
    st.floats(min_value=0.05, max_value=0.5),
    st.floats(min_value=0.5, max_value=0.99),
)
def test_mondrian_quantile_monotone_in_one_minus_alpha(
    pairs: list[tuple[float, float]], a1: float, a2: float
) -> None:
    if a1 == a2:
        return
    if a1 > a2:
        a1, a2 = a2, a1
    preds = np.asarray([p for p, _ in pairs])
    labs = np.asarray([y for _, y in pairs])
    mc = MondrianConformal.fit({"g": preds}, {"g": labs}, min_group_size=2)
    q1 = mc.quantile("g", alpha=a1)
    q2 = mc.quantile("g", alpha=a2)
    if math.isfinite(q1) and math.isfinite(q2):
        assert q1 >= q2 - 1e-12


class _RandomNLI:
    """Stub backend that returns hash-derived entailment, so semantic entropy
    runs end-to-end on a hypothesis-generated input."""

    async def classify_pair(self, premise: str, hypothesis: str) -> NLIScores:
        same = hash((premise, hypothesis)) % 3 == 0
        return NLIScores(
            entailment=0.9 if same else 0.1,
            neutral=0.05,
            contradiction=0.05 if same else 0.85,
            top=NLILabel.ENTAILMENT if same else NLILabel.CONTRADICTION,
        )

    async def classify_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> list[NLIScores]:
        results: list[NLIScores] = []
        for p, h in pairs:
            results.append(await self.classify_pair(p, h))
        return results

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_semantic_entropy_bounds() -> None:
    samples = [f"sample-{i}" for i in range(8)]
    result = await semantic_entropy(samples, nli=_RandomNLI())
    assert 0.0 <= result.entropy <= max_entropy(len(samples)) + 1e-9


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=50,
    ),
)
def test_self_consistency_ci_bracket_nonnegative(samples: list[float]) -> None:
    result = self_consistency_stats(
        np.asarray(samples), alpha=0.1, n_bootstrap=300, rng=np.random.default_rng(0)
    )
    assert result.ci_high >= result.ci_low - 1e-12
    assert result.variance >= 0.0
    assert result.iqr >= 0.0
