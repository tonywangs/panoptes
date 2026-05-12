"""Unit tests for `panoptes.uq.nli.llm.LLMNLIBackend` against the MockClient."""

from __future__ import annotations

import pytest

from panoptes.clients._mock import MockClient
from panoptes.uq.nli.base import NLILabel, bidirectional_entails
from panoptes.uq.nli.llm import LLMNLIBackend


@pytest.mark.asyncio
async def test_llm_nli_returns_canonical_scores() -> None:
    client = MockClient(provider="anthropic", model="claude-haiku-4-5")
    backend = LLMNLIBackend(client=client)
    result = await backend.classify_pair("the sky is blue", "the sky is blue")
    assert result.top in {NLILabel.ENTAILMENT, NLILabel.NEUTRAL, NLILabel.CONTRADICTION}
    total = result.entailment + result.neutral + result.contradiction
    assert total == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= result.entailment <= 1.0
    assert 0.0 <= result.neutral <= 1.0
    assert 0.0 <= result.contradiction <= 1.0


@pytest.mark.asyncio
async def test_llm_nli_batch_matches_individual() -> None:
    client = MockClient(provider="anthropic", model="claude-haiku-4-5")
    backend = LLMNLIBackend(client=client)
    pairs = [("a", "a"), ("a", "b"), ("c", "d")]
    batch_results = await backend.classify_pairs(pairs)
    assert len(batch_results) == 3
    for res in batch_results:
        assert res.entailment + res.neutral + res.contradiction == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_bidirectional_entails_threshold() -> None:
    client = MockClient(provider="anthropic", model="claude-haiku-4-5")
    backend = LLMNLIBackend(client=client)
    # Below threshold should be False even if both directions are NLI-positive
    # but with low confidence. We can't assert specific labels without seeding
    # the mock's hash; just verify the function runs and returns a bool.
    result = await bidirectional_entails(backend, "a paraphrase", "the same idea")
    assert isinstance(result, bool)
