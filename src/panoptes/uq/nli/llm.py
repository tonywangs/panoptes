"""LLM-as-NLI backend.

When the local HF backend isn't available (no GPU, no `providers-hf` extra,
or simply not wanted), fall back to asking an LLM via structured output.
This costs O(N²) calls per item per (judge,sampling) batch but introduces
no extra Python deps.

The judging LLM is asked to label the relationship between a *premise* and a
*hypothesis* as `entailment`, `neutral`, or `contradiction`, with a
`confidence` ∈ [0, 1]. We convert that to `NLIScores` by placing the
confidence on the chosen label and dividing the remainder uniformly across
the other two — a simple temperature-flattened pseudo-probability that
preserves the argmax and roughly tracks calibration.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from panoptes.clients._structured import (
    force_tool_choice,
    model_to_tool_spec,
    parse_tool_call,
)
from panoptes.clients.base import LLMClient, Message, SystemBlock
from panoptes.uq.nli.base import NLILabel, NLIScores

_TOOL_NAME = "record_nli"
_TOOL_DESCRIPTION = (
    "Label the logical relationship between premise and hypothesis as "
    "'entailment' (premise entails hypothesis), 'neutral' (neither entails "
    "nor contradicts), or 'contradiction'. Provide a confidence in [0, 1]."
)
_SYSTEM_PROMPT = (
    "You are a careful natural-language-inference annotator. Given a "
    "premise and a hypothesis, decide whether the premise entails the "
    "hypothesis, is neutral toward it, or contradicts it. Two paraphrases "
    "of the same factual claim are entailment in both directions. Two "
    "statements with the same surface form but different referents are "
    "not entailment. Call the record_nli tool exactly once."
)


class _NLIToolPayload(BaseModel):
    """Structured-output schema the LLM-as-NLI tool returns."""

    label: Literal["entailment", "neutral", "contradiction"]
    confidence: float = Field(ge=0.0, le=1.0)


class LLMNLIBackend:
    """`NLIBackend` impl that delegates each pair classification to an LLM."""

    def __init__(self, *, client: LLMClient, max_tokens: int = 128) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._tool = model_to_tool_spec(
            _NLIToolPayload, name=_TOOL_NAME, description=_TOOL_DESCRIPTION
        )

    async def classify_pair(self, premise: str, hypothesis: str) -> NLIScores:
        user = (
            f"[Premise]\n{premise}\n\n[Hypothesis]\n{hypothesis}\n\n"
            "Classify the relationship."
        )
        completion = await self._client.complete(
            messages=[Message(role="user", content=user)],
            system=[SystemBlock(text=_SYSTEM_PROMPT, cache_control="ephemeral")],
            tools=[self._tool],
            tool_choice=force_tool_choice(_TOOL_NAME),
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        payload = parse_tool_call(completion, tool_name=_TOOL_NAME, schema=_NLIToolPayload)
        return _to_nli_scores(payload)

    async def classify_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> list[NLIScores]:
        # No batching in the underlying API; the pipeline-level concurrency
        # (per-provider Semaphore) handles parallelism.
        results: list[NLIScores] = []
        for premise, hypothesis in pairs:
            results.append(await self.classify_pair(premise, hypothesis))
        return results

    async def aclose(self) -> None:
        await self._client.aclose()


def _to_nli_scores(payload: _NLIToolPayload) -> NLIScores:
    """Convert the LLM's single label+confidence into a 3-way score vector.

    The chosen label takes the LLM's confidence; the remaining mass is split
    evenly across the two other labels. This is a coarse approximation but
    preserves the argmax and is monotone in the LLM's reported confidence.
    """
    label = NLILabel(payload.label)
    p_top = payload.confidence
    p_rest = (1.0 - p_top) / 2.0
    entailment = p_top if label is NLILabel.ENTAILMENT else p_rest
    neutral = p_top if label is NLILabel.NEUTRAL else p_rest
    contradiction = p_top if label is NLILabel.CONTRADICTION else p_rest
    return NLIScores(
        entailment=entailment,
        neutral=neutral,
        contradiction=contradiction,
        top=label,
    )
