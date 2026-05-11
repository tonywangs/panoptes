"""Error taxonomy for PANOPTES.

The framework distinguishes *retriable* errors (transient network conditions,
rate limits, provider overload) from *terminal* errors (auth failure, malformed
request, validation error). Only retriable errors are retried with backoff;
terminal errors surface immediately so the user notices misconfiguration.

The mapping from HTTP status to retriable/terminal follows common practice:
    - 408, 409 (conflict), 425, 429, 5xx -> retriable
    - 400, 401, 403, 404, 422 -> terminal
Provider-specific overloaded signals (e.g. Anthropic's `overloaded_error`) are
forced to retriable regardless of status code.
"""

from __future__ import annotations


class PanoptesError(Exception):
    """Root of the PANOPTES error hierarchy."""


class ConfigError(PanoptesError):
    """Misconfiguration detected at startup (missing API key, bad path, etc)."""


class BenchmarkError(PanoptesError):
    """Failure loading, parsing, or iterating a benchmark dataset."""


class JudgeError(PanoptesError):
    """A judge call returned a structurally invalid response (e.g. schema parse failure).

    This is distinct from `LLMClientError` subclasses: those describe transport-level
    failures, this describes a successful API response whose contents cannot be
    coerced into the judge's expected structured-output schema.
    """


class StorageError(PanoptesError):
    """duckdb / prompt-cache layer failure."""


class LLMClientError(PanoptesError):
    """Base class for LLM client transport / API errors.

    Carries an optional HTTP status code and, where the provider supplies one,
    a `retry_after_s` hint from `Retry-After` or equivalent.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.provider = provider


class RetriableError(LLMClientError):
    """Transient failure: 429, 5xx, network timeout, provider overload.

    The retry loop will back off (exponential + jitter) and try again until
    `max_retries` is exhausted, after which the final attempt's exception
    is re-raised to the caller as-is.
    """


class TerminalError(LLMClientError):
    """Non-retriable failure: 4xx auth/validation, schema-incompatible request.

    Surfaces immediately. Never retried — retrying would only burn quota and
    delay the eventual failure.
    """


def classify_http_status(status: int) -> type[RetriableError | TerminalError]:
    """Map an HTTP status code to the appropriate error class.

    Conservative: anything not explicitly retriable is treated as terminal.
    Provider-specific signals (e.g. Anthropic overloaded_error at 529) should
    be handled in the provider client before this function is consulted.
    """
    if status in {408, 409, 425, 429} or 500 <= status <= 599:
        return RetriableError
    return TerminalError
