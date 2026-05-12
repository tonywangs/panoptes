"""`NLIBackend` Protocol + shared types.

Semantic entropy clusters temperature-sampled responses by *bidirectional
entailment*: two responses belong to the same cluster iff each entails the
other. The NLI backend is the swappable component that decides entailment;
PANOPTES ships two: a local HF DeBERTa-v3-mnli classifier
(`nli/deberta.py`) and an LLM-as-NLI fallback (`nli/llm.py`).

We standardize on three labels: `entailment`, `neutral`, `contradiction`.
Some pretrained MNLI checkpoints use different label orderings; concrete
backends are responsible for mapping their model's labels onto our enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class NLILabel(StrEnum):
    """Canonical NLI labels."""

    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True, slots=True)
class NLIScores:
    """Probability over the three canonical labels for one (premise, hypothesis).

    Probabilities should sum to ~1.0 (numerical tolerance allowed). `top` is
    the argmax label, broken out for convenience.
    """

    entailment: float
    neutral: float
    contradiction: float
    top: NLILabel

    def probability(self, label: NLILabel) -> float:
        if label is NLILabel.ENTAILMENT:
            return self.entailment
        if label is NLILabel.NEUTRAL:
            return self.neutral
        return self.contradiction


@runtime_checkable
class NLIBackend(Protocol):
    """Anything that classifies a premise/hypothesis pair into NLI labels.

    Implementations should accept arbitrary natural-language strings and
    return `NLIScores`. Batching is strongly encouraged but not required;
    `classify_pairs` defaults to a naive loop over `classify_pair`.
    """

    async def classify_pair(self, premise: str, hypothesis: str) -> NLIScores: ...

    async def classify_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> list[NLIScores]: ...

    async def aclose(self) -> None: ...


async def bidirectional_entails(
    backend: NLIBackend,
    a: str,
    b: str,
    *,
    threshold: float = 0.5,
) -> bool:
    """Return True iff `a` entails `b` AND `b` entails `a` above `threshold`.

    The default threshold of 0.5 matches the Farquhar et al. (2024) protocol:
    a pair counts as "mutually entailing" when both directions' entailment
    probability exceeds 0.5. Backends with sharper classifiers can call with
    a higher threshold (e.g. 0.7) for tighter clusters.
    """
    forward, backward = await backend.classify_pairs([(a, b), (b, a)])
    return forward.entailment >= threshold and backward.entailment >= threshold
