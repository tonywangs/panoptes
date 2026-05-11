"""Provider-agnostic async LLM client Protocol + retry / cost machinery.

The public surface is:
    - `LLMClient`: the Protocol every provider implements
    - `Message`, `SystemBlock`, `ToolSpec`, `ToolChoice`, `CompletionResponse`:
        provider-neutral request/response value types
    - `with_backoff`: shared retry decorator that distinguishes retriable from
        terminal errors, applies exponential backoff with full jitter, and
        respects provider `Retry-After` hints
    - `price_call`: USD cost lookup for a given (model, usage) pair

Design notes:
    - The Protocol intentionally exposes a minimal surface. Anthropic's
        prompt-caching is handled by setting `cache_control` markers on
        `SystemBlock` entries; OpenAI's cached_tokens are auto-detected from
        the response usage. Provider-specific subtleties live in each
        concrete client, not in the Protocol.
    - Semaphores are created on demand per provider so a process with only
        an Anthropic key doesn't allocate quota for unused providers.
    - We hand-roll backoff (rather than depend on tenacity) so the retry
        policy is transparent in the codebase and easy to audit. Retries
        are bounded; non-retriable errors bypass the loop entirely.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from panoptes.errors import LLMClientError, RetriableError, TerminalError
from panoptes.schemas import TokenUsage


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in the model conversation. `content` may be plain text or a
    list of typed blocks for image/tool-use messages (handled by each provider)."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class SystemBlock:
    """A system-prompt chunk. `cache_control` is an Anthropic concept and is a
    no-op for other providers, but kept on the value type so the pipeline does
    not need provider-conditional code paths."""

    text: str
    cache_control: Literal["none", "ephemeral"] = "none"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A single tool definition exposed to the model.

    `input_schema` is a JSON Schema (object) describing valid `input` payloads.
    For structured-output enforcement, judges declare a single tool with
    `tool_choice = ToolChoice(type='tool', name=...)`.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """How aggressively to force tool use.

    - 'auto':   model may or may not call a tool
    - 'any':    model must call some tool
    - 'tool':   model must call `name` specifically (used for structured output)
    """

    type: Literal["auto", "any", "tool"] = "auto"
    name: str | None = None


@dataclass(slots=True)
class CompletionResponse:
    """One model reply, normalized across providers.

    `tool_calls` is keyed by tool name with parsed JSON-object inputs.
    Structured-output judges read from `tool_calls[<tool_name>]` rather than
    parsing `text`, so judge code is robust to providers that interleave
    text and tool-use blocks.
    """

    text: str
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict[str, Any])


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic async chat-completion Protocol.

    Implementations are responsible for: (a) translating these neutral
    request types into the provider's wire format, (b) parsing the response
    into a `CompletionResponse`, (c) wrapping retriable failures in
    `RetriableError` and terminal failures in `TerminalError`, and
    (d) populating `cost_usd` via `price_call`.
    """

    provider: str
    model: str

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
    ) -> CompletionResponse: ...

    async def aclose(self) -> None: ...


async def with_backoff[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_s: float,
    max_s: float,
    rng: random.Random | None = None,
) -> T:
    """Run `fn` with exponential backoff + full jitter on `RetriableError`.

    Terminal errors propagate immediately. The final attempt's exception is
    re-raised as-is to preserve the original status code and traceback. If
    a `RetriableError` carries a `retry_after_s` hint (from `Retry-After`
    headers), it overrides the computed backoff for that attempt.

    Backoff schedule:
        delay = min(max_s, base_s * 2^attempt) * uniform(0, 1)
    This is the "full jitter" variant from the AWS architecture blog
    (https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/),
    which has the lowest expected completion time among standard variants.
    """
    rand = rng if rng is not None else random.Random()
    last_exc: LLMClientError | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except TerminalError:
            raise
        except RetriableError as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            if exc.retry_after_s is not None:
                delay = exc.retry_after_s
            else:
                cap = min(max_s, base_s * (2**attempt))
                delay = cap * rand.random()
            await asyncio.sleep(delay)
    # Unreachable: the loop above either returns, raises, or sleeps; the final
    # iteration's `raise` is always taken when retries are exhausted. We keep
    # this guard for the type checker.
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Per-provider concurrency control
# ---------------------------------------------------------------------------

_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def get_semaphore(provider: str, max_concurrency: int) -> asyncio.Semaphore:
    """Return (creating if absent) the asyncio.Semaphore for `provider`.

    Sized at `max_concurrency` for the first caller; subsequent calls return
    the existing instance regardless of the size argument. This is deliberate:
    a global cap per provider beats letting multiple call sites contend.
    """
    sem = _SEMAPHORES.get(provider)
    if sem is None:
        sem = asyncio.Semaphore(max_concurrency)
        _SEMAPHORES[provider] = sem
    return sem


def reset_semaphores() -> None:
    """Drop the per-provider semaphore registry. Test-only."""
    _SEMAPHORES.clear()


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


# USD per 1M tokens. Values are best-effort current public list prices; update
# when providers change them. Cache read/write fields are Anthropic-specific
# (OpenAI absorbs cache pricing into `input_tokens_cached` at a discount which
# we approximate by treating cache_read at 0.1x input rate).
@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


_PRICING: dict[str, ModelPricing] = {
    # Anthropic
    "claude-opus-4-7": ModelPricing(15.0, 75.0, 1.5, 18.75),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0, 0.3, 3.75),
    "claude-haiku-4-5": ModelPricing(1.0, 5.0, 0.1, 1.25),
    # OpenAI (illustrative)
    "gpt-4o": ModelPricing(2.5, 10.0, 1.25, 0.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.6, 0.075, 0.0),
    # Google (illustrative)
    "gemini-2.5-pro": ModelPricing(1.25, 10.0, 0.0, 0.0),
}


def price_call(model: str, usage: TokenUsage) -> float:
    """Compute USD cost for a single call given normalized `TokenUsage`.

    Returns 0.0 for unknown models with a single best-effort fallback: pretend
    the model is priced like sonnet (a middle-of-the-pack default). This means
    cost reports for unrecognized models are *indicative*, not precise — the
    user should register pricing in `_PRICING` for production accounting.
    """
    pricing = _PRICING.get(model)
    if pricing is None:
        pricing = _PRICING["claude-sonnet-4-6"]
    return (
        pricing.input_per_mtok * usage.input_tokens
        + pricing.output_per_mtok * usage.output_tokens
        + pricing.cache_read_per_mtok * usage.cache_read_tokens
        + pricing.cache_write_per_mtok * usage.cache_creation_tokens
    ) / 1_000_000.0


def register_pricing(model: str, pricing: ModelPricing) -> None:
    """Register (or override) pricing for a model. Useful for new SKUs."""
    _PRICING[model] = pricing
