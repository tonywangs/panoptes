"""Unit tests for `panoptes.uq.semantic_entropy`.

Uses a stubbed `NLIBackend` that returns deterministic entailment based on
string equality (or a user-supplied policy), so we can verify the
clustering and entropy math without an actual NLI model.
"""

from __future__ import annotations

import math

import pytest

from panoptes.uq.nli.base import NLILabel, NLIScores
from panoptes.uq.semantic_entropy import max_entropy, semantic_entropy


class _StringEqualityNLI:
    """`NLIBackend` that returns entailment iff the strings are identical."""

    async def classify_pair(self, premise: str, hypothesis: str) -> NLIScores:
        same = premise == hypothesis
        return NLIScores(
            entailment=1.0 if same else 0.0,
            neutral=0.0,
            contradiction=0.0 if same else 1.0,
            top=NLILabel.ENTAILMENT if same else NLILabel.CONTRADICTION,
        )

    async def classify_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> list[NLIScores]:
        results: list[NLIScores] = []
        for premise, hypothesis in pairs:
            results.append(await self.classify_pair(premise, hypothesis))
        return results

    async def aclose(self) -> None:
        return None


class _AllSameNLI:
    """`NLIBackend` that returns entailment for every pair."""

    async def classify_pair(self, premise: str, hypothesis: str) -> NLIScores:
        del premise, hypothesis
        return NLIScores(
            entailment=1.0, neutral=0.0, contradiction=0.0, top=NLILabel.ENTAILMENT
        )

    async def classify_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> list[NLIScores]:
        return [
            NLIScores(
                entailment=1.0, neutral=0.0, contradiction=0.0, top=NLILabel.ENTAILMENT
            )
            for _ in pairs
        ]

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_all_distinct_samples_give_log_n_entropy() -> None:
    samples = ["alpha", "beta", "gamma", "delta"]
    result = await semantic_entropy(samples, nli=_StringEqualityNLI())
    assert result.n_clusters == 4
    assert result.entropy == pytest.approx(math.log(4), abs=1e-9)


@pytest.mark.asyncio
async def test_identical_samples_give_zero_entropy() -> None:
    samples = ["x", "x", "x", "x"]
    result = await semantic_entropy(samples, nli=_AllSameNLI())
    assert result.n_clusters == 1
    assert result.entropy == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_entropy_within_bounds() -> None:
    samples = ["a", "a", "b", "c", "b"]
    result = await semantic_entropy(samples, nli=_StringEqualityNLI())
    assert 0.0 <= result.entropy <= max_entropy(len(samples)) + 1e-9


@pytest.mark.asyncio
async def test_log_probs_weighting_changes_entropy() -> None:
    samples = ["a", "a", "b", "c"]
    nli = _StringEqualityNLI()
    uniform = await semantic_entropy(samples, nli=nli)
    # Concentrate weight on cluster 'a' via log-probs.
    weighted = await semantic_entropy(
        samples, nli=nli, log_probs=[0.0, 0.0, -10.0, -10.0]
    )
    # Concentrating mass on one cluster strictly lowers entropy.
    assert weighted.entropy < uniform.entropy


def test_max_entropy_helper() -> None:
    assert max_entropy(1) == 0.0
    assert max_entropy(2) == pytest.approx(math.log(2))
    assert max_entropy(10) == pytest.approx(math.log(10))


@pytest.mark.asyncio
async def test_rejects_singleton_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        await semantic_entropy(["only one"], nli=_StringEqualityNLI())
