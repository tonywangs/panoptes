"""Example: add a new judge in one file.

Run with:
    uv run python examples/custom_judge.py

This script defines `LengthAwareRubricJudge`, a small variant of the
standard rubric judge that *also* records the response's character
length in the rationale. It demonstrates the M5 acceptance criterion
"a new contributor can add a new judge by implementing one Protocol
class and registering it; no other code changes required".

The judge satisfies `panoptes.judges.base.Judge` by exposing:
    - `judge_id: str`
    - `async evaluate(item, model_response, *, sample_index, temperature) -> JudgeResponse`

That's it. The pipeline only depends on the Protocol, so plugging this
in is a one-import operation.
"""

from __future__ import annotations

import asyncio
import time

from panoptes.clients._mock import MockClient
from panoptes.clients._structured import (
    force_tool_choice,
    model_to_tool_spec,
    parse_tool_call,
)
from panoptes.clients.base import LLMClient, Message, SystemBlock
from panoptes.schemas import BenchmarkItem, JudgeResponse, RubricScore, TaskFamily

_TOOL_NAME = "record_score"


class LengthAwareRubricJudge:
    """A custom judge that scores via the standard rubric but also flags
    long responses (> 1000 chars) via the rubric's `flags` field."""

    def __init__(self, *, client: LLMClient, judge_id: str) -> None:
        self._client = client
        self.judge_id = judge_id
        self._tool = model_to_tool_spec(
            RubricScore, name=_TOOL_NAME, description="Record score."
        )

    async def evaluate(
        self,
        item: BenchmarkItem,
        model_response: str,
        *,
        sample_index: int = 0,
        temperature: float | None = None,
    ) -> JudgeResponse:
        t0 = time.perf_counter()
        effective_temp = 0.0 if temperature is None else temperature
        completion = await self._client.complete(
            messages=[
                Message(
                    role="user",
                    content=f"Score this response (1-line OK):\n{model_response[:500]}",
                )
            ],
            system=[SystemBlock(text="You are a length-aware code judge.")],
            tools=[self._tool],
            tool_choice=force_tool_choice(_TOOL_NAME),
            max_tokens=256,
            temperature=effective_temp,
        )
        score = parse_tool_call(completion, tool_name=_TOOL_NAME, schema=RubricScore)
        # Augment with our custom signal:
        if len(model_response) > 1000:
            score = score.model_copy(update={"flags": [*score.flags, "long_response"]})
        return JudgeResponse(
            judge_id=self.judge_id,
            item_id=item.item_id,
            score=score,
            raw_text=completion.text,
            usage=completion.usage,
            cost_usd=completion.cost_usd,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            prompt_hash="custom-length-aware-v1",
            sampled_at_temperature=effective_temp,
            sample_index=sample_index,
        )


async def _main() -> None:
    client = MockClient(provider="example", model="example-model")
    judge = LengthAwareRubricJudge(
        client=client, judge_id="example:example-model:length-aware-v1"
    )
    item = BenchmarkItem(
        item_id="demo/1",
        benchmark="demo",
        task_family=TaskFamily.CODE,
        prompt="def add(a, b): ...",
        reference=None,
    )
    response = await judge.evaluate(item, "def add(a, b):\n    return a + b\n")
    print(f"judge_id      : {response.judge_id}")
    print(f"score.value   : {response.score.value:.3f}")
    print(f"score.flags   : {response.score.flags}")
    print(f"cost_usd      : ${response.cost_usd:.6f}")


if __name__ == "__main__":
    asyncio.run(_main())
