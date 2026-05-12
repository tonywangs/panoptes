"""Unit tests for `panoptes.clients.openai` via respx-mocked HTTP."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from panoptes.clients.base import Message, reset_semaphores
from panoptes.clients.openai import OpenAIClient
from panoptes.errors import RetriableError, TerminalError


@pytest.fixture(autouse=True)
def _reset_sems() -> None:  # pyright: ignore[reportUnusedFunction]
    reset_semaphores()


def _ok_payload(score: float = 0.8) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "record_score",
                                "arguments": json.dumps(
                                    {
                                        "value": score,
                                        "scale": "continuous",
                                        "rationale": "test",
                                        "flags": [],
                                    }
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 50},
        },
    }


@pytest.mark.asyncio
@respx.mock
async def test_openai_happy_path_parses_tool_call_and_caches() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_payload(0.42))
    )
    client = OpenAIClient(api_key="k", model="gpt-4o", max_retries=0)
    try:
        result = await client.complete(messages=[Message(role="user", content="hi")])
    finally:
        await client.aclose()
    assert result.tool_calls["record_score"]["value"] == 0.42
    # 200 prompt - 50 cached = 150 fresh input, 50 cache-read.
    assert result.usage.input_tokens == 150
    assert result.usage.cache_read_tokens == 50


@pytest.mark.asyncio
@respx.mock
async def test_openai_401_terminal() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401, json={"error": {"type": "invalid_api_key", "message": "bad key"}}
        )
    )
    client = OpenAIClient(api_key="bad", model="gpt-4o", max_retries=5)
    try:
        with pytest.raises(TerminalError):
            await client.complete(messages=[Message(role="user", content="x")])
    finally:
        await client.aclose()
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_openai_429_retries_with_retry_after() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"type": "rate_limit_error", "message": "slow down"}},
            headers={"retry-after": "0.01"},
        )
    )
    client = OpenAIClient(api_key="k", model="gpt-4o", max_retries=0)
    try:
        with pytest.raises(RetriableError) as exc:
            await client.complete(messages=[Message(role="user", content="x")])
    finally:
        await client.aclose()
    assert exc.value.retry_after_s == pytest.approx(0.01)
