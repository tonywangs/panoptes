"""Single-call rubric judge: take a (task, response) pair, return a RubricScore.

This is the workhorse judge for M1. It:
    1. Loads a versioned prompt template.
    2. Sends `[system, user]` to the underlying `LLMClient` with the
       `record_score` tool forced via `ToolChoice(type='tool', name=...)`.
    3. Parses the tool-call payload into a `RubricScore`.
    4. Wraps everything in a `JudgeResponse` with cost + usage attached.

The system block is marked with `cache_control='ephemeral'` so Anthropic's
prompt-cache can amortize the rubric across the dataset; the user block,
which contains the per-item task and candidate response, is *not* cached.
For providers that ignore `cache_control`, the marker is a no-op.
"""

from __future__ import annotations

from panoptes.clients._structured import (
    force_tool_choice,
    model_to_tool_spec,
    parse_tool_call,
)
from panoptes.clients.base import LLMClient, Message, SystemBlock
from panoptes.judges.base import PromptTemplate
from panoptes.schemas import BenchmarkItem, JudgeResponse, RubricScore

_TOOL_NAME = "record_score"
_TOOL_DESCRIPTION = (
    "Record a scalar quality score in [0.0, 1.0] together with a short "
    "rationale and any applicable concern flags. Call exactly once."
)


class RubricJudge:
    """Concrete judge that scores via a rubric template + forced tool call.

    The `judge_id` is constructed at init time from the provider, model, and
    a `variant` label (typically the prompt version). It is the canonical
    way the framework refers to this judge in storage, routing, and the bandit.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        template: PromptTemplate,
        variant: str,
        default_temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._template = template
        self._default_temperature = default_temperature
        self._max_tokens = max_tokens
        self.judge_id = f"{client.provider}:{client.model}:{variant}"
        self._tool = model_to_tool_spec(
            RubricScore,
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
        )

    async def evaluate(
        self,
        item: BenchmarkItem,
        model_response: str,
        *,
        sample_index: int = 0,
        temperature: float | None = None,
    ) -> JudgeResponse:
        effective_temp = temperature if temperature is not None else self._default_temperature
        user_text = self._template.render_user(
            prompt=item.prompt,
            response=model_response,
        )
        completion = await self._client.complete(
            messages=[Message(role="user", content=user_text)],
            system=[SystemBlock(text=self._template.system, cache_control="ephemeral")],
            tools=[self._tool],
            tool_choice=force_tool_choice(_TOOL_NAME),
            max_tokens=self._max_tokens,
            temperature=effective_temp,
        )
        score = parse_tool_call(completion, tool_name=_TOOL_NAME, schema=RubricScore)
        return JudgeResponse(
            judge_id=self.judge_id,
            item_id=item.item_id,
            score=score,
            raw_text=completion.text,
            usage=completion.usage,
            cost_usd=completion.cost_usd,
            latency_ms=completion.latency_ms,
            prompt_hash=self._template.content_hash,
            sampled_at_temperature=effective_temp,
            sample_index=sample_index,
        )
