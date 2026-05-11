"""End-to-end evaluation pipeline.

The CLI is a thin wrapper around `run_evaluation`. Tests drive this function
directly with a mock client. Two passes are run per item:

1. **Point pass** (`temperature=0`): one call per judge. Skipped per-item if
   `PromptCache` reports a hit. Results are written to duckdb immediately so
   crashes don't lose work.
2. **Sampling pass** (`temperature>0`, `n_samples>1`): no-op in M1; the
   sampling-based UQ methods (semantic entropy, self-consistency) land in M2.

Concurrency: judge calls fan out via `asyncio.gather` per item; the per-
provider `asyncio.Semaphore` inside each client caps simultaneous requests.
This gives us "everything in flight, bounded by provider quota" without the
pipeline knowing each provider's quota directly.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from panoptes.judges.base import Judge
from panoptes.schemas import (
    BenchmarkItem,
    ConformalResult,
    CostReport,
    EvalRecord,
    JudgeResponse,
    JuryDecision,
)
from panoptes.storage.duckdb_store import DuckDBStore, row_from_response
from panoptes.storage.prompt_cache import PromptCache
from panoptes.uq.conformal_split import SplitConformal

log = logging.getLogger(__name__)


@dataclass(slots=True)
class EvalConfig:
    """Knobs the pipeline reads. Build from CLI args; tests construct directly."""

    run_id: str
    alpha: float = 0.1
    uq_methods: tuple[str, ...] = ("split",)
    skip_cached: bool = True


@dataclass(slots=True)
class JudgeRef:
    """Pairs a judge with the prompt template hash it uses.

    The pipeline groups results by `judge.judge_id`; the hash is forwarded
    to the prompt-cache layer so we can ask "has this judge already scored
    this item under this rubric version?".
    """

    judge: Judge
    prompt_version_hash: str


async def run_evaluation(
    *,
    items: Sequence[BenchmarkItem],
    responses: dict[str, str],
    judges: Sequence[JudgeRef],
    store: DuckDBStore,
    cache: PromptCache | None,
    config: EvalConfig,
    model_under_test: str,
) -> CostReport:
    """Run the M1 point-pass eval. Returns a per-judge cost summary.

    Errors from individual judge calls propagate. We deliberately do not
    swallow failures: a single failing judge usually means a config bug
    (missing key, schema mismatch) the user must see immediately.
    """
    store.record_run(
        run_id=config.run_id,
        config={
            "alpha": config.alpha,
            "uq_methods": list(config.uq_methods),
            "judges": [ref.judge.judge_id for ref in judges],
            "n_items": len(items),
            "model_under_test": model_under_test,
        },
    )
    cost = _CostAccumulator()
    all_records: list[EvalRecord] = []

    for item in items:
        response_text = responses.get(item.item_id)
        if response_text is None:
            log.warning("No response provided for item %s; skipping", item.item_id)
            continue
        record = EvalRecord(
            run_id=config.run_id,
            item=item,
            model_under_test=model_under_test,
            model_response=response_text,
        )
        responses_per_judge = await _judge_item(item, response_text, judges, cache, config)
        for jr in responses_per_judge:
            record.judge_responses.append(jr)
            cost.add(jr)
        if record.judge_responses:
            decision = _aggregate_point(record.judge_responses)
            if "split" in config.uq_methods:
                conformal = _fit_split_for_decision(record.judge_responses, alpha=config.alpha)
                if conformal is not None:
                    decision.conformal.append(conformal)
            record.jury_decision = decision
        all_records.append(record)
        store.write_rows(row_from_response(record, jr) for jr in record.judge_responses)

    log.info("Pipeline finished: %d items, %d judge calls", len(all_records), cost.n_calls)
    return cost.to_report()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _judge_item(
    item: BenchmarkItem,
    response_text: str,
    judges: Sequence[JudgeRef],
    cache: PromptCache | None,
    config: EvalConfig,
) -> list[JudgeResponse]:
    """Fan out to all judges in parallel; respect prompt-cache if configured."""

    async def _one(ref: JudgeRef) -> JudgeResponse | None:
        if (
            cache is not None
            and config.skip_cached
            and cache.is_cached(
                benchmark=item.benchmark,
                judge_id=ref.judge.judge_id,
                item_id=item.item_id,
                prompt_version_hash=ref.prompt_version_hash,
            )
        ):
            log.debug("Cache hit for %s / %s", item.item_id, ref.judge.judge_id)
            return None
        return await ref.judge.evaluate(item, response_text)

    results = await asyncio.gather(*[_one(ref) for ref in judges])
    return [r for r in results if r is not None]


def _aggregate_point(responses: list[JudgeResponse]) -> JuryDecision:
    """Simple mean aggregator for M1. Replaced in M3 by hierarchical Gaussian."""
    values = np.asarray([r.score.value for r in responses], dtype=np.float64)
    mean = float(values.mean())
    var = float(values.var(ddof=1)) if len(values) > 1 else 0.0
    return JuryDecision(
        item_id=responses[0].item_id,
        posterior_mean=mean,
        posterior_var=var,
        judges_called=[r.judge_id for r in responses],
        cost_usd=sum(r.cost_usd for r in responses),
        strategy="all",
    )


def _fit_split_for_decision(
    responses: list[JudgeResponse], *, alpha: float
) -> ConformalResult | None:
    """Build a split-conformal interval around the mean using inter-judge spread.

    This is a stand-in calibration for M1: residuals are |score_j - mean|.
    With ground-truth labels (M5 calibration probe), this is replaced by
    proper held-out calibration. We emit the result so the storage and
    reporting paths see a real `ConformalResult` shape.
    """
    if len(responses) < 2:
        return None
    values = np.asarray([r.score.value for r in responses], dtype=np.float64)
    mean = float(values.mean())
    residuals = np.abs(values - mean)
    cp = SplitConformal(residuals=residuals)
    lo, hi = cp.predict_interval(mean, alpha=alpha)
    return ConformalResult(method="split", alpha=alpha, point=mean, lo=lo, hi=hi)


@dataclass(slots=True)
class _CostAccumulator:
    n_calls: int = 0
    n_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    usd_total: float = 0.0
    by_judge: dict[str, float] = field(default_factory=dict[str, float])

    def add(self, response: JudgeResponse) -> None:
        self.n_calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.cache_read += response.usage.cache_read_tokens
        self.cache_write += response.usage.cache_creation_tokens
        self.usd_total += response.cost_usd
        self.by_judge[response.judge_id] = (
            self.by_judge.get(response.judge_id, 0.0) + response.cost_usd
        )

    def to_report(self) -> CostReport:
        return CostReport(
            usd_total=self.usd_total,
            by_judge=dict(self.by_judge),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read,
            cache_creation_tokens=self.cache_write,
            n_calls=self.n_calls,
            n_retries=self.n_retries,
        )


def new_run_id() -> str:
    """Generate a run id of the form `panoptes-<8 hex>`."""
    return f"panoptes-{uuid.uuid4().hex[:8]}"
