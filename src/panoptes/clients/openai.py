"""OpenAI Chat Completions API client.

Implements `LLMClient` against
    POST https://api.openai.com/v1/chat/completions

Notable differences from Anthropic:
    - tool definitions are nested under `{"type": "function", "function": {...}}`
    - `tool_choice` for a specific tool uses
        `{"type": "function", "function": {"name": "..."}}`
    - usage fields are `prompt_tokens`/`completion_tokens` (not
        `input_tokens`/`output_tokens`); `prompt_tokens_details.cached_tokens`
        maps to our `cache_read_tokens`
    - tool-call `arguments` arrive as a JSON-encoded *string*, not a dict
    - there is no Anthropic-style `cache_control` to inject; cache is
        provider-managed

`system` blocks (Anthropic concept) are mapped onto a leading
`{"role": "system", "content": ...}` message; multiple SystemBlock entries
are joined with newlines because Chat Completions takes a single system
message.
"""

from __future__ import annotations

import json
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
from panoptes.errors import JudgeError, RetriableError, TerminalError, classify_http_status
from panoptes.schemas import TokenUsage

_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIClient:
    """Concrete `LLMClient` for OpenAI's Chat Completions API.

    Construct with an explicit API key and model. `base_url` is configurable
    so the same class powers `openai_compat.py` (Together, Groq, vLLM) with
    only a base-url override.
    """

    provider: str = "openai"

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
        base_url: str = _OPENAI_BASE_URL,
        provider_id: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        if provider_id is not None:
            self.provider = provider_id
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff_base = backoff_base_s
        self._backoff_max = backoff_max_s
        self._extra_headers = dict(extra_headers or {})
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
                    f"{self._base_url}/chat/completions",
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
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        return headers

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
        chat_messages: list[dict[str, Any]] = []
        if system is not None and len(system) > 0:
            chat_messages.append(
                {"role": "system", "content": "\n\n".join(b.text for b in system)}
            )
        for m in messages:
            chat_messages.append({"role": m.role, "content": m.content})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools is not None and len(tools) > 0:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": dict(t.input_schema),
                    },
                }
                for t in tools
            ]
        if tool_choice is not None:
            body["tool_choice"] = _encode_tool_choice(tool_choice)
        if stop_sequences:
            body["stop"] = stop_sequences
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

        cls = classify_http_status(status)
        retry_after = _parse_retry_after(response.headers.get("retry-after"))
        if cls is RetriableError:
            raise RetriableError(
                f"{self.provider} {status} {err_type}: {message}",
                status_code=status,
                retry_after_s=retry_after,
                provider=self.provider,
            )
        raise TerminalError(
            f"{self.provider} {status} {err_type}: {message}",
            status_code=status,
            provider=self.provider,
        )

    def _parse(self, payload: dict[str, Any], *, latency_ms: float) -> CompletionResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:  # pyright: ignore[reportUnknownArgumentType]
            raise JudgeError(f"{self.provider} response missing choices: {payload!r}")
        first = choices[0]
        if not isinstance(first, dict):
            raise JudgeError(f"{self.provider} response choice malformed: {first!r}")
        message_obj = first.get("message", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(message_obj, dict):
            raise JudgeError(f"{self.provider} response missing message: {first!r}")
        content = message_obj.get("content")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        text = content if isinstance(content, str) else ""
        tool_calls: dict[str, dict[str, Any]] = {}
        raw_tool_calls = message_obj.get("tool_calls", [])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(raw_tool_calls, list):
            for call in raw_tool_calls:  # pyright: ignore[reportUnknownVariableType]
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                args = fn.get("arguments")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if not isinstance(name, str):
                    continue
                if isinstance(args, str):
                    try:
                        parsed_args = json.loads(args)
                    except json.JSONDecodeError as exc:
                        raise JudgeError(
                            f"{self.provider} tool '{name}' arguments not valid JSON: {args!r}"
                        ) from exc
                elif isinstance(args, dict):
                    parsed_args = args
                else:
                    raise JudgeError(
                        f"{self.provider} tool '{name}' arguments unexpected type: {type(args).__name__}"
                    )
                if isinstance(parsed_args, dict):
                    tool_calls[name] = {str(k): v for k, v in parsed_args.items()}  # type: ignore[redundant-cast]
        usage_raw = payload.get("usage", {})
        if not isinstance(usage_raw, dict):
            usage_raw = {}
        details = usage_raw.get("prompt_tokens_details", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        cached = 0
        if isinstance(details, dict):
            cached_raw = details.get("cached_tokens", 0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(cached_raw, int):
                cached = cached_raw
        prompt_tokens = int(usage_raw.get("prompt_tokens", 0))  # pyright: ignore[reportArgumentType]
        completion_tokens = int(usage_raw.get("completion_tokens", 0))  # pyright: ignore[reportArgumentType]
        usage = TokenUsage(
            input_tokens=max(0, prompt_tokens - cached),
            output_tokens=completion_tokens,
            cache_read_tokens=cached,
            cache_creation_tokens=0,
        )
        cost = price_call(self.model, usage)
        return CompletionResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw=payload,
        )


def _encode_tool_choice(choice: ToolChoice) -> Any:
    if choice.type == "tool":
        if choice.name is None:
            raise ValueError("ToolChoice(type='tool') requires a name")
        return {"type": "function", "function": {"name": choice.name}}
    if choice.type == "any":
        return "required"
    return "auto"


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
