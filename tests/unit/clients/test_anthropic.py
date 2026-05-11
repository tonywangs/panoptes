"""Unit tests for `panoptes.clients.anthropic` via respx-mocked HTTP.

These tests stand in for hitting the real Anthropic API: respx intercepts
httpx calls and returns canned responses, so we can verify:

    - happy-path tool-use parsing & usage extraction
    - 429 is classified `RetriableError` with `retry_after_s`
    - 401 is classified `TerminalError` (no retry)
    - 529 / overloaded_error is `RetriableError`
    - backoff retries succeed when a transient 429 is followed by 200
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from panoptes.clients.anthropic import AnthropicClient
from panoptes.clients.base import Message, reset_semaphores
from panoptes.errors import RetriableError, TerminalError


@pytest.fixture(autouse=True)
def _reset_sems() -> None:  # pyright: ignore[reportUnusedFunction]
    """Drop the per-provider semaphore registry between tests."""
    reset_semaphores()


def _ok_payload(score: float = 0.9) -> dict[str, object]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "record_score",
                "input": {
                    "value": score,
                    "scale": "continuous",
                    "rationale": "looks fine",
                    "flags": [],
                },
            }
        ],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 25,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 0,
        },
    }


@pytest.mark.asyncio
@respx.mock
async def test_happy_path_parses_tool_use_and_usage() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_ok_payload(0.77))
    )
    client = AnthropicClient(
        api_key="test-key",
        model="claude-sonnet-4-6",
        max_retries=0,
    )
    try:
        result = await client.complete(
            messages=[Message(role="user", content="hello")],
            max_tokens=64,
        )
    finally:
        await client.aclose()
    assert result.tool_calls["record_score"]["value"] == 0.77
    assert result.usage.input_tokens == 120
    assert result.usage.cache_read_tokens == 50
    assert result.cost_usd > 0.0


@pytest.mark.asyncio
@respx.mock
async def test_429_is_retriable_with_retry_after() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"type": "rate_limit_error", "message": "slow down"}},
            headers={"retry-after": "1.5"},
        )
    )
    client = AnthropicClient(
        api_key="test-key",
        model="claude-sonnet-4-6",
        max_retries=0,
    )
    try:
        with pytest.raises(RetriableError) as excinfo:
            await client.complete(messages=[Message(role="user", content="x")])
    finally:
        await client.aclose()
    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after_s == pytest.approx(1.5)


@pytest.mark.asyncio
@respx.mock
async def test_401_is_terminal_and_not_retried() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"type": "authentication_error", "message": "bad key"}},
        )
    )
    client = AnthropicClient(
        api_key="bad",
        model="claude-sonnet-4-6",
        max_retries=5,
    )
    try:
        with pytest.raises(TerminalError) as excinfo:
            await client.complete(messages=[Message(role="user", content="x")])
    finally:
        await client.aclose()
    assert excinfo.value.status_code == 401
    # Should not retry: exactly one HTTP call despite max_retries=5.
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_529_overloaded_classified_retriable() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            529,
            json={"error": {"type": "overloaded_error", "message": "busy"}},
        )
    )
    client = AnthropicClient(
        api_key="k",
        model="claude-sonnet-4-6",
        max_retries=0,
    )
    try:
        with pytest.raises(RetriableError):
            await client.complete(messages=[Message(role="user", content="x")])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_backoff_retries_then_succeeds() -> None:
    payload = _ok_payload(0.5)
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"type": "rate_limit_error", "message": "x"}}),
            httpx.Response(200, json=payload),
        ]
    )
    client = AnthropicClient(
        api_key="k",
        model="claude-sonnet-4-6",
        max_retries=2,
        backoff_base_s=0.001,
        backoff_max_s=0.002,
    )
    try:
        result = await asyncio.wait_for(
            client.complete(messages=[Message(role="user", content="x")]),
            timeout=5.0,
        )
    finally:
        await client.aclose()
    assert route.call_count == 2
    assert result.tool_calls["record_score"]["value"] == 0.5
