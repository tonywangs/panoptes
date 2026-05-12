"""Unit tests for `panoptes.clients.google` via respx-mocked HTTP."""

from __future__ import annotations

import httpx
import pytest
import respx

from panoptes.clients.base import Message, reset_semaphores
from panoptes.clients.google import GoogleClient
from panoptes.errors import RetriableError, TerminalError


@pytest.fixture(autouse=True)
def _reset_sems() -> None:  # pyright: ignore[reportUnusedFunction]
    reset_semaphores()


def _ok_payload(score: float = 0.7) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "record_score",
                                "args": {
                                    "value": score,
                                    "scale": "continuous",
                                    "rationale": "ok",
                                    "flags": [],
                                },
                            }
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 180,
            "candidatesTokenCount": 35,
        },
    }


@pytest.mark.asyncio
@respx.mock
async def test_google_happy_path() -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    ).mock(return_value=httpx.Response(200, json=_ok_payload(0.55)))
    client = GoogleClient(api_key="k", model="gemini-2.5-pro", max_retries=0)
    try:
        result = await client.complete(messages=[Message(role="user", content="hi")])
    finally:
        await client.aclose()
    assert result.tool_calls["record_score"]["value"] == 0.55
    assert result.usage.input_tokens == 180
    assert result.usage.output_tokens == 35


@pytest.mark.asyncio
@respx.mock
async def test_google_403_terminal() -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    ).mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "code": 403,
                    "status": "PERMISSION_DENIED",
                    "message": "forbidden",
                }
            },
        )
    )
    client = GoogleClient(api_key="bad", model="gemini-2.5-pro", max_retries=2)
    try:
        with pytest.raises(TerminalError):
            await client.complete(messages=[Message(role="user", content="x")])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_google_503_retriable() -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    ).mock(
        return_value=httpx.Response(
            503,
            json={
                "error": {"code": 503, "status": "UNAVAILABLE", "message": "busy"}
            },
        )
    )
    client = GoogleClient(api_key="k", model="gemini-2.5-pro", max_retries=0)
    try:
        with pytest.raises(RetriableError):
            await client.complete(messages=[Message(role="user", content="x")])
    finally:
        await client.aclose()
