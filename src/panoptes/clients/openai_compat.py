"""OpenAI-compatible HTTP clients: Together, Groq, local vLLM.

These providers expose the same Chat Completions / tools schema as OpenAI at
different base URLs. We subclass `OpenAIClient` and override `provider`,
`base_url`, and `extra_headers` as appropriate. Pricing falls back to the
default if a Together / Groq model isn't in `_PRICING`; users wanting
accurate accounting should register their model via
`panoptes.clients.base.register_pricing`.

Known base URLs:
    - Together: https://api.together.xyz/v1
    - Groq:     https://api.groq.com/openai/v1
    - vLLM:     user-configurable (e.g. http://localhost:8000/v1)
"""

from __future__ import annotations

import httpx

from panoptes.clients.openai import OpenAIClient
from panoptes.config import RateLimit

_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class TogetherClient(OpenAIClient):
    """Together.ai OpenAI-compatible client."""

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
        base_url: str = _TOGETHER_BASE_URL,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            rate_limit=rate_limit,
            request_timeout_s=request_timeout_s,
            connect_timeout_s=connect_timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
            http_client=http_client,
            base_url=base_url,
            provider_id="together",
        )


class GroqClient(OpenAIClient):
    """Groq OpenAI-compatible client."""

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
        base_url: str = _GROQ_BASE_URL,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            rate_limit=rate_limit,
            request_timeout_s=request_timeout_s,
            connect_timeout_s=connect_timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
            http_client=http_client,
            base_url=base_url,
            provider_id="groq",
        )


class VLLMClient(OpenAIClient):
    """Local vLLM OpenAI-compatible client.

    The user supplies `base_url` (e.g. `http://localhost:8000/v1`). `api_key`
    defaults to the dummy string `"EMPTY"` since vLLM ignores it but still
    requires an Authorization header.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        rate_limit: RateLimit | None = None,
        request_timeout_s: float = 60.0,
        connect_timeout_s: float = 10.0,
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            rate_limit=rate_limit,
            request_timeout_s=request_timeout_s,
            connect_timeout_s=connect_timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
            http_client=http_client,
            base_url=base_url,
            provider_id="vllm",
        )
