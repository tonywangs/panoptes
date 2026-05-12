"""Google Gemini `generateContent` client.

Implements `LLMClient` against
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent

Gemini's wire format differs from OpenAI/Anthropic enough that this client
does not share much with `openai.py`:

    - messages are `contents: [{role: 'user'|'model', parts: [{text: ...}]}]`
      (note: assistant role is named `model`, not `assistant`)
    - system prompts are `system_instruction: {parts: [{text: ...}]}`
    - tools are `tools: [{function_declarations: [{name, description, parameters}]}]`
    - to force a specific function, set
      `tool_config.function_calling_config: {mode: 'ANY', allowed_function_names: ['...']}`
    - usage is `usageMetadata.promptTokenCount` / `candidatesTokenCount`
    - API key is passed as `?key=...` query parameter (deprecated Bearer is
      also accepted but we follow the documented convention)
    - error envelope is `{"error": {"code": int, "status": str, "message": str}}`

Gemini does not yet expose a cache-read field analogous to Anthropic's; we
populate `cache_read_tokens=0` and document this in the cost report.
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
from panoptes.errors import JudgeError, RetriableError, TerminalError, classify_http_status
from panoptes.schemas import TokenUsage

_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleClient:
    """Concrete `LLMClient` for Google's Gemini `generateContent`."""

    provider: str = "google"

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
        base_url: str = _GOOGLE_BASE_URL,
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
        body = self._build_body(
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        url = f"{self._base_url}/models/{self.model}:generateContent"

        async def _do() -> CompletionResponse:
            async with self._sem:
                t0 = time.perf_counter()
                response = await self._http.post(
                    url,
                    params={"key": self._api_key},
                    headers={"Content-Type": "application/json"},
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
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = "user" if m.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system is not None and len(system) > 0:
            body["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(b.text for b in system)}],
            }
        if tools is not None and len(tools) > 0:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": dict(t.input_schema),
                        }
                        for t in tools
                    ]
                }
            ]
        if tool_choice is not None:
            body["toolConfig"] = {
                "functionCallingConfig": _encode_function_calling_config(tool_choice)
            }
        if stop_sequences:
            body["generationConfig"]["stopSequences"] = stop_sequences
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
        err_status = str(error_obj.get("status", "UNKNOWN"))
        message = str(error_obj.get("message", response.text))
        cls = classify_http_status(status)
        retry_after = _parse_retry_after(response.headers.get("retry-after"))
        if cls is RetriableError:
            raise RetriableError(
                f"google {status} {err_status}: {message}",
                status_code=status,
                retry_after_s=retry_after,
                provider=self.provider,
            )
        raise TerminalError(
            f"google {status} {err_status}: {message}",
            status_code=status,
            provider=self.provider,
        )

    def _parse(self, payload: dict[str, Any], *, latency_ms: float) -> CompletionResponse:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or len(candidates) == 0:  # pyright: ignore[reportUnknownArgumentType]
            raise JudgeError(f"google response missing candidates: {payload!r}")
        first = candidates[0]
        if not isinstance(first, dict):
            raise JudgeError(f"google candidate malformed: {first!r}")
        content = first.get("content", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        text_chunks: list[str] = []
        tool_calls: dict[str, dict[str, Any]] = {}
        if isinstance(content, dict):
            parts = content.get("parts", [])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(parts, list):
                for part in parts:  # pyright: ignore[reportUnknownVariableType]
                    if not isinstance(part, dict):
                        continue
                    if "text" in part:
                        text_val = part["text"]  # pyright: ignore[reportUnknownVariableType]
                        if isinstance(text_val, str):
                            text_chunks.append(text_val)
                    if "functionCall" in part:
                        call = part["functionCall"]  # pyright: ignore[reportUnknownVariableType]
                        if isinstance(call, dict):
                            name = call.get("name")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                            args = call.get("args", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                            if isinstance(name, str) and isinstance(args, dict):
                                tool_calls[name] = {str(k): v for k, v in args.items()}  # type: ignore[redundant-cast]
        usage_raw = payload.get("usageMetadata", {})
        if not isinstance(usage_raw, dict):
            usage_raw = {}
        input_tokens = int(usage_raw.get("promptTokenCount", 0))  # pyright: ignore[reportArgumentType]
        output_tokens = int(usage_raw.get("candidatesTokenCount", 0))  # pyright: ignore[reportArgumentType]
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )
        cost = price_call(self.model, usage)
        return CompletionResponse(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw=payload,
        )


def _encode_function_calling_config(choice: ToolChoice) -> dict[str, Any]:
    if choice.type == "tool":
        if choice.name is None:
            raise ValueError("ToolChoice(type='tool') requires a name")
        return {"mode": "ANY", "allowedFunctionNames": [choice.name]}
    if choice.type == "any":
        return {"mode": "ANY"}
    return {"mode": "AUTO"}


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
