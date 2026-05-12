"""Example: add a new LLM provider in one file.

Run with:
    uv run python examples/custom_provider.py

This file implements `EchoClient`, a toy `LLMClient` that doesn't call any
network service — it simply echoes a deterministic structured response
matching whatever tool was requested. It demonstrates that adding a new
provider is one Protocol implementation; no other PANOPTES code changes.

The contract is in `panoptes.clients.base.LLMClient`:

    class LLMClient(Protocol):
        provider: str
        model: str
        async def complete(self, messages, *, system=None, tools=None,
                           tool_choice=None, max_tokens=1024, temperature=0.0,
                           stop_sequences=None) -> CompletionResponse: ...
        async def aclose(self) -> None: ...

Implement those and you're done. The dispatch in `cli.py:_build_client` is
the only place that needs to know about the new alias, and that's a
config change (one line in `_JUDGE_ALIASES`), not a code change.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from panoptes.clients.base import (
    CompletionResponse,
    LLMClient,
    Message,
    SystemBlock,
    ToolChoice,
    ToolSpec,
    price_call,
)
from panoptes.schemas import TokenUsage


@dataclass(slots=True)
class EchoClient:
    """A no-network `LLMClient` that returns a constant `RubricScore` payload."""

    provider: str = "echo"
    model: str = "echo-1"
    fixed_score: float = 0.75

    async def complete(
        self,
        messages: list[Message],
        *,
        system: list[SystemBlock] | None = None,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop_sequences: list[str] | None = None,
    ) -> CompletionResponse:
        del system, tool_choice, max_tokens, temperature, stop_sequences
        tool_name = tools[0].name if tools else "record_score"
        prompt_text = "\n".join(m.content for m in messages)
        usage = TokenUsage(input_tokens=len(prompt_text) // 4, output_tokens=20)
        return CompletionResponse(
            text="",
            tool_calls={
                tool_name: {
                    "value": self.fixed_score,
                    "scale": "continuous",
                    "rationale": "echo: constant score",
                    "flags": [],
                }
            },
            usage=usage,
            cost_usd=price_call(self.model, usage),
            latency_ms=0.5,
            raw={"_echo": True},
        )

    async def aclose(self) -> None:
        return None


async def _main() -> None:
    client: LLMClient = EchoClient(fixed_score=0.42)
    result = await client.complete(
        messages=[Message(role="user", content="hi")],
        tools=[
            ToolSpec(
                name="record_score",
                description="Record a score in [0,1].",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                },
            )
        ],
    )
    print(f"provider     : {client.provider}")
    print(f"model        : {client.model}")
    print(f"value        : {result.tool_calls['record_score']['value']}")
    print(f"usage.input  : {result.usage.input_tokens}")
    print(f"cost_usd     : ${result.cost_usd:.6f}")


if __name__ == "__main__":
    asyncio.run(_main())
