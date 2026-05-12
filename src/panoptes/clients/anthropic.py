"""Anthropic Messages API client.

Implements `LLMClient` via `httpx.AsyncClient` against
    POST https://api.anthropic.com/v1/messages

Features:
    - tool-use structured output (single forced tool for judges)
    - prompt caching via `cache_control: {"type": "ephemeral"}` markers on
        system blocks
    - usage parsing including `cache_read_input_tokens` and
        `cache_creation_input_tokens`
    - retriable/terminal error classification, with `Retry-After` honored on
        429s

The official `anthropic` SDK would also work, but a hand-written httpx client
gives us:
    - direct control over retry semantics (the SDK's retry is opinionated),
    - identical concurrency control via shared `asyncio.Semaphore`,
    - smaller dependency footprint (no `httpx-sse`, no `tokenizers`).

We rely on the public `anthropic-version: 2023-06-01` API surface and
`anthropic-beta: prompt-caching-2024-07-31` for prompt-cache headers, both of
which Anthropic has committed to keeping stable.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from panoptes.clients.base import (
    CompletionResponse,
    Message,
    SystemBlock,
    ToolChoice,
    ToolSpec,
    get_semaphore,
    price_call,
    with_backoff,
)
from panoptes.config import RateLimit
from panoptes.errors import RetriableError, TerminalError, classify_http_status
from panoptes.schemas import TokenUsage

_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_API_VERSION = "2023-06-01"
_ANTHROPIC_BETAS = "prompt-caching-2024-07-31"


class AnthropicClient:
    """Concrete `LLMClient` for Anthropic's Messages API.

    Construct with an explicit API key and model identifier. The instance
    owns an `httpx.AsyncClient`; call `aclose()` to release it (or use it as
    an async context manager in higher-level pipeline code).
    """

    provider: str = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        rate_limit: RateLimit | None = None,
        request_timeout_s: float = 60.0,
        connect_timeout_s: float = 10.0,
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = _ANTHROPIC_BASE_URL,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff_base = backoff_base_s
        self._backoff_max = backoff_max_s
        if http_client is None:
            timeout = httpx.Timeout(
                connect=connect_timeout_s,
                read=request_timeout_s,
                write=request_timeout_s,
                pool=request_timeout_s,
            )
            self._http = httpx.AsyncClient(timeout=timeout)
            self._owns_http = True
        else:
            self._http = http_client
            self._owns_http = False
        limit = rate_limit if rate_limit is not None else RateLimit()
        self._sem = get_semaphore(self.provider, limit.max_concurrency)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

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
        """Send one Messages-API request, with backoff on retriable errors."""
        body = self._build_body(
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

        async def _do() -> CompletionResponse:
            async with self._sem:
                t0 = time.perf_counter()
                response = await self._http.post(
                    f"{self._base_url}/messages",
                    headers=self._headers(),
                    json=body,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
            self._raise_for_status(response)
            return self._parse(response.json(), latency_ms=latency_ms)

        return await with_backoff(
            _do,
            max_retries=self._max_retries,
            base_s=self._backoff_base,
            max_s=self._backoff_max,
        )

    # ------------------------------------------------------------------ helpers

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "anthropic-beta": _ANTHROPIC_BETAS,
            "content-type": "application/json",
        }

    def _build_body(
        self,
        *,
        messages: list[Message],
        system: list[SystemBlock] | None,
        tools: list[ToolSpec] | None,
        tool_choice: ToolChoice | None,
        max_tokens: int,
        temperature: float,
        stop_sequences: list[str] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if system is not None and len(system) > 0:
            body["system"] = [_encode_system_block(b) for b in system]
        if tools is not None and len(tools) > 0:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": dict(t.input_schema),
                }
                for t in tools
            ]
        if tool_choice is not None:
            body["tool_choice"] = _encode_tool_choice(tool_choice)
        if stop_sequences:
            body["stop_sequences"] = stop_sequences
        return body

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        status = response.status_code
        error_obj: dict[str, Any] = {}
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            maybe_err = parsed.get("error")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(maybe_err, dict):
                error_obj = {str(k): v for k, v in maybe_err.items()}  # type: ignore[redundant-cast]
        err_type = str(error_obj.get("type", "unknown_error"))
        message = str(error_obj.get("message", response.text))
        # Anthropic's `overloaded_error` is retriable regardless of status code
        # (in practice it ships as 529, but classify defensively).
        if err_type == "overloaded_error" or status == 529:
            raise RetriableError(
                f"anthropic overloaded: {message}",
                status_code=status,
                provider=self.provider,
            )
        cls = classify_http_status(status)
        retry_after = _parse_retry_after(response.headers.get("retry-after"))
        if cls is RetriableError:
            raise RetriableError(
                f"anthropic {status} {err_type}: {message}",
                status_code=status,
                retry_after_s=retry_after,
                provider=self.provider,
            )
        raise TerminalError(
            f"anthropic {status} {err_type}: {message}",
            status_code=status,
            provider=self.provider,
        )

    def _parse(self, payload: dict[str, Any], *, latency_ms: float) -> CompletionResponse:
        text_chunks: list[str] = []
        tool_calls: dict[str, dict[str, Any]] = {}
        for block in payload.get("content", []):
            btype = block.get("type")
            if btype == "text":
                text_chunks.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                if isinstance(inp, dict):
                    tool_calls[name] = {str(k): v for k, v in inp.items()}  # type: ignore[redundant-cast]
        usage_raw = payload.get("usage", {})
        usage = TokenUsage(
            input_tokens=int(usage_raw.get("input_tokens", 0)),
            output_tokens=int(usage_raw.get("output_tokens", 0)),
            cache_read_tokens=int(usage_raw.get("cache_read_input_tokens", 0)),
            cache_creation_tokens=int(usage_raw.get("cache_creation_input_tokens", 0)),
        )
        cost = price_call(_normalize_model_for_pricing(self.model), usage)
        return CompletionResponse(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw=payload,
        )


def _encode_system_block(block: SystemBlock) -> dict[str, Any]:
    encoded: dict[str, Any] = {"type": "text", "text": block.text}
    if block.cache_control == "ephemeral":
        encoded["cache_control"] = {"type": "ephemeral"}
    return encoded


def _encode_tool_choice(choice: ToolChoice) -> dict[str, Any]:
    if choice.type == "tool":
        if choice.name is None:
            raise ValueError("ToolChoice(type='tool') requires a name")
        return {"type": "tool", "name": choice.name}
    return {"type": choice.type}


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _normalize_model_for_pricing(model: str) -> str:
    """Strip the optional date suffix Anthropic appends to model identifiers.

    Example: 'claude-sonnet-4-6-20251022' -> 'claude-sonnet-4-6'. If the
    suffix isn't present the input is returned unchanged. Failing to strip
    just means we fall back to the default price in `price_call`.
    """
    parts = model.split("-")
    if parts and parts[-1].isdigit() and len(parts[-1]) == 8:
        return "-".join(parts[:-1])
    return model
