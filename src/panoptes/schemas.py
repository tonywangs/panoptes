"""Pydantic v2 schemas for the PANOPTES public surface.

Every cross-module data structure passes through this file. Schemas are
designed for:
    - JSON serialization (storage in duckdb sidecar columns, network transport),
    - structured-output enforcement (judges return RubricScore via tool-use),
    - pyright --strict compatibility (no Any leakage in public types).

Frozen models are used for value types that should be hashable and immutable
(`BenchmarkItem`, `TokenUsage`, `CostReport`). Mutable container types
(`EvalRecord`, `JuryDecision`) accumulate state during a run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    field_validator,
)


def _now_utc() -> datetime:
    """Tiny helper so default_factory types are inferable by pyright in strict mode."""
    return datetime.now(UTC)


class TaskFamily(StrEnum):
    """Top-level partition of the benchmark space.

    Used as the Mondrian-conformal grouping key and as a coordinate in the
    duckdb partition layout. Add new families sparingly — each one fragments
    the calibration set.
    """

    CODE = "code"
    MATH = "math"
    FACTUALITY = "factuality"
    FREEFORM = "freeform"


class ScoreScale(StrEnum):
    """How a rubric score is to be interpreted statistically.

    CONTINUOUS scores live in [0, 1] and are aggregated with the hierarchical
    Gaussian model. LIKERT_1_5 scores are ordinal categorical and aggregated
    with ordinal Dawid-Skene.
    """

    CONTINUOUS = "continuous"
    LIKERT_1_5 = "likert_1_5"


class BenchmarkItem(BaseModel):
    """One example drawn from a benchmark.

    `metadata` is intentionally untyped (JSON object) — benchmark-specific
    fields (HumanEval's `test`, GSM8K's `answer`, etc.) live here so the core
    pipeline doesn't need to know about each benchmark's idiosyncrasies.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(description="Stable unique id within the benchmark.")
    benchmark: str = Field(description="Benchmark name, e.g. 'humaneval'.")
    task_family: TaskFamily
    prompt: str = Field(description="The task prompt presented to the model under eval.")
    reference: str | None = Field(
        default=None,
        description="Ground-truth answer when available (verifiable benchmarks).",
    )
    metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict[str, str | int | float | bool | None]
    )


class RubricScore(BaseModel):
    """Structured output schema returned by every rubric judge.

    The `value` field is on a unit interval regardless of the underlying scale,
    so downstream UQ code can treat all judges uniformly. The original `scale`
    is preserved so aggregation can choose the right model (continuous Gaussian
    vs ordinal DS). For LIKERT_1_5, value = (likert - 1) / 4.
    """

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, le=1.0, description="Score on the unit interval.")
    scale: ScoreScale = ScoreScale.CONTINUOUS
    likert: int | None = Field(default=None, ge=1, le=5)
    rationale: str = Field(description="Judge's brief justification.")
    flags: list[str] = Field(
        default_factory=list[str],
        description="Optional concerns the judge surfaced (e.g. 'off_topic', 'ambiguous_prompt').",
    )

    @field_validator("scale", mode="before")
    @classmethod
    def _normalize_scale(cls, v: Any) -> Any:
        """LLMs frequently emit the enum *name* (`'LIKERT_1_5'`) rather than the
        *value* (`'likert_1_5'`) even when the JSON Schema enum lists the
        lowercase value. Lowercase any string input before enum coercion so
        we don't lose an entire run to a casing quirk in one judge call.
        """
        if isinstance(v, str):
            return v.lower()
        return v


class TokenUsage(BaseModel):
    """Per-call token accounting; provider-normalized.

    `cache_read` and `cache_creation` are Anthropic-specific concepts; other
    providers will populate 0 unless they expose equivalent fields. OpenAI's
    `cached_tokens` is mapped to `cache_read`.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cache_read_tokens: NonNegativeInt = 0
    cache_creation_tokens: NonNegativeInt = 0


class JudgeResponse(BaseModel):
    """One judge's answer for one item.

    `prompt_hash` is the git-tracked content hash of the rubric template; rerunning
    with an unchanged hash hits the cache. `latency_ms` is wall time including
    backoff/retries, useful for the bandit's information-per-second variant.
    """

    judge_id: str
    item_id: str
    score: RubricScore
    raw_text: str = Field(description="The judge's raw model output, pre-parse.")
    usage: TokenUsage
    cost_usd: NonNegativeFloat
    latency_ms: NonNegativeFloat
    prompt_hash: str
    sampled_at_temperature: float = Field(
        ge=0.0,
        description="0.0 for point-estimate pass; >0 for sampling-based UQ.",
    )
    sample_index: int = Field(
        default=0,
        ge=0,
        description="When n>1 samples are drawn at temp>0, this is their index.",
    )
    timestamp_utc: datetime = Field(default_factory=_now_utc)


class ConformalResult(BaseModel):
    """A prediction interval produced by one conformal method.

    `method` is the discriminator for downstream analysis; `alpha` is the
    nominal miscoverage rate (so the targeted coverage is 1 - alpha).
    """

    method: Literal["split", "adaptive", "mondrian"]
    alpha: float = Field(gt=0.0, lt=1.0)
    point: float = Field(ge=0.0, le=1.0)
    lo: float = Field(ge=0.0, le=1.0)
    hi: float = Field(ge=0.0, le=1.0)
    group: str | None = Field(
        default=None,
        description="Mondrian group key when applicable, else None.",
    )


class UncertaintyDecomposition(BaseModel):
    """Aleatoric/epistemic split of total predictive variance.

    Computed via nested resampling: outer bootstrap over judges (epistemic),
    inner over temperature samples within judge (aleatoric). See `uq/decomposition.py`.
    """

    model_config = ConfigDict(frozen=True)

    total: NonNegativeFloat
    aleatoric: NonNegativeFloat
    epistemic: NonNegativeFloat
    n_judges: NonNegativeInt
    n_samples_per_judge: NonNegativeInt


class JuryDecision(BaseModel):
    """Aggregated multi-judge verdict for one item."""

    item_id: str
    posterior_mean: float = Field(ge=0.0, le=1.0)
    posterior_var: NonNegativeFloat
    decomposition: UncertaintyDecomposition | None = None
    conformal: list[ConformalResult] = Field(default_factory=list[ConformalResult])
    judges_called: list[str]
    cost_usd: NonNegativeFloat
    strategy: Literal["all", "single", "escalation", "bandit"] = "all"


class CostReport(BaseModel):
    """Per-run cost summary, broken down by judge.

    `by_judge` keys are the same `judge_id` strings used elsewhere
    (e.g. 'anthropic:claude-sonnet-4-6:rubric-v1').
    """

    model_config = ConfigDict(frozen=True)

    usd_total: NonNegativeFloat
    by_judge: dict[str, NonNegativeFloat] = Field(default_factory=dict[str, NonNegativeFloat])
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cache_read_tokens: NonNegativeInt = 0
    cache_creation_tokens: NonNegativeInt = 0
    n_calls: NonNegativeInt = 0
    n_retries: NonNegativeInt = 0


class EvalRecord(BaseModel):
    """One row's worth of state: the eval pipeline's unit of work.

    Flows through the pipeline as: created with item + model_response, then
    accumulates judge_responses, then `jury_decision`. Persisted to duckdb
    by `storage.duckdb_store.write_records`.
    """

    run_id: str
    item: BenchmarkItem
    model_under_test: str = Field(description="The model whose output is being judged.")
    model_response: str
    judge_responses: list[JudgeResponse] = Field(default_factory=list[JudgeResponse])
    jury_decision: JuryDecision | None = None
    created_at_utc: datetime = Field(default_factory=_now_utc)


JudgeId = Annotated[
    str,
    Field(
        pattern=r"^[a-z0-9_\-]+:[a-zA-Z0-9_.\-]+:[a-zA-Z0-9_\-]+$",
        description="Format: 'provider:model:judge_variant', e.g. "
        "'anthropic:claude-sonnet-4-6:rubric-code-v1'.",
    ),
]
