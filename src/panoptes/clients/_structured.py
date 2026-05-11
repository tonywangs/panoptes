"""Adapters between Pydantic models and provider tool-use schemas.

Every judge declares a Pydantic model as its output schema. We compile that
model to a JSON Schema once, attach it as a single-tool `ToolSpec`, and force
the provider to emit the structured payload via `ToolChoice(type='tool', ...)`.

This module is the only place that knows how to round-trip
    `Pydantic model class` <-> `JSON Schema dict`
so providers do not each reinvent it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from panoptes.clients.base import CompletionResponse, ToolChoice, ToolSpec
from panoptes.errors import JudgeError


def model_to_tool_spec(
    model: type[BaseModel],
    *,
    name: str,
    description: str,
) -> ToolSpec:
    """Wrap a Pydantic model class as a single-purpose tool definition.

    The model's JSON Schema is used verbatim as `input_schema`. Anthropic and
    OpenAI both accept JSON Schema draft-7 with `properties`, `required`, and
    `additionalProperties`; Pydantic v2 emits a compatible dialect.
    """
    schema = model.model_json_schema()
    # Anthropic rejects schemas with $defs at the top level for tool inputs;
    # Pydantic only emits $defs when there are nested models. For M1 the rubric
    # model is flat, but we strip defensively in case a future judge nests.
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    return ToolSpec(name=name, description=description, input_schema=schema)


def force_tool_choice(tool_name: str) -> ToolChoice:
    """Convenience: build a ToolChoice that demands `tool_name` be invoked."""
    return ToolChoice(type="tool", name=tool_name)


def parse_tool_call[T: BaseModel](
    response: CompletionResponse,
    *,
    tool_name: str,
    schema: type[T],
) -> T:
    """Extract a typed Pydantic instance from `response.tool_calls[tool_name]`.

    Raises `JudgeError` if the tool was not invoked or the payload fails
    Pydantic validation. We surface validation errors with the offending
    payload because debugging silent miscoercion is painful.
    """
    payload = response.tool_calls.get(tool_name)
    if payload is None:
        raise JudgeError(
            f"Judge response missing required tool call '{tool_name}'. "
            f"Got text-only output of length {len(response.text)}."
        )
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise JudgeError(
            f"Judge tool-call payload failed schema validation for {schema.__name__}: "
            f"{exc.errors()!r}; payload was {payload!r}"
        ) from exc


def coerce_payload(raw: object) -> dict[str, Any]:
    """Best-effort coercion of a provider's tool-use input into a plain dict.

    Anthropic returns `input` as a JSON object already; OpenAI returns a string
    that must be JSON-loaded. Subclasses pre-normalize before storing into
    `CompletionResponse.tool_calls`, so this helper is mostly defensive.
    """
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}  # type: ignore[redundant-cast]
    raise JudgeError(f"Unexpected tool input shape: {type(raw).__name__}")
