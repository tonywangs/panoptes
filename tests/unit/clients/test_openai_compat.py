"""Smoke tests for the OpenAI-compat subclasses.

These confirm that `TogetherClient` and `GroqClient` hit their correct base
URLs and reuse the parent's parser. Detailed protocol behavior is covered
by `test_openai.py`.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from panoptes.clients.base import Message, reset_semaphores
from panoptes.clients.openai_compat import GroqClient, TogetherClient


@pytest.fixture(autouse=True)
def _reset_sems() -> None:  # pyright: ignore[reportUnusedFunction]
    reset_semaphores()


def _ok_payload() -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "record_score",
                                "arguments": json.dumps(
                                    {
                                        "value": 0.5,
                                        "scale": "continuous",
                                        "rationale": "x",
                                        "flags": [],
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


@pytest.mark.asyncio
@respx.mock
async def test_together_hits_together_base_url() -> None:
    route = respx.post("https://api.together.xyz/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_payload())
    )
    client = TogetherClient(api_key="k", model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
    try:
        result = await client.complete(messages=[Message(role="user", content="hi")])
    finally:
        await client.aclose()
    assert route.call_count == 1
    assert client.provider == "together"
    assert result.tool_calls["record_score"]["value"] == 0.5


@pytest.mark.asyncio
@respx.mock
async def test_groq_hits_groq_base_url() -> None:
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_payload())
    )
    client = GroqClient(api_key="k", model="llama-3.1-70b-versatile")
    try:
        result = await client.complete(messages=[Message(role="user", content="hi")])
    finally:
        await client.aclose()
    assert route.call_count == 1
    assert client.provider == "groq"
    assert result.tool_calls["record_score"]["value"] == 0.5
