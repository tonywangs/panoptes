"""End-to-end evaluation pipeline.

The CLI is a thin wrapper around `run_evaluation`. Tests drive this function
directly with a mock client. Two passes are run per item:

1. **Point pass** (`temperature=0`): one call per judge. Skipped per-item if
   `PromptCache` reports a hit. Results are written to duckdb immediately so
   crashes don't lose work.
2. **Sampling pass** (`temperature=temperature_sampling`, `n_samples > 0`):
   one call per (judge, item, sample_index) at the sampling temperature.
   Used by `self-consistency` and `semantic-entropy` UQ methods. Sampling
   responses are stored with `sample_index > 0` and bypass the prompt cache
   (each sample is an independent draw).

Concurrency: judge calls fan out via `asyncio.gather` per item; the per-
provider `asyncio.Semaphore` inside each client caps simultaneous requests.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from panoptes.judges.base import Judge
from panoptes.routing.base import JudgeCatalog, JudgeMeta, JuryRouter
from panoptes.routing.strategies import AllJudges
from panoptes.schemas import (
    BenchmarkItem,
    ConformalResult,
    CostReport,
    EvalRecord,
    JudgeResponse,
    JuryDecision,
    UncertaintyDecomposition,
)
from panoptes.storage.duckdb_store import DuckDBStore, row_from_response
from panoptes.storage.prompt_cache import PromptCache
from panoptes.uq.conformal_split import SplitConformal
from panoptes.uq.decomposition import decompose_variance
from panoptes.uq.nli.base import NLIBackend
from panoptes.uq.self_consistency import self_consistency_stats
from panoptes.uq.semantic_entropy import semantic_entropy

log = logging.getLogger(__name__)

_SAMPLING_UQ_METHODS = frozenset({"self-consistency", "semantic-entropy"})


@dataclass(slots=True)
class EvalConfig:
    """Knobs the pipeline reads. Build from CLI args; tests construct directly."""

    run_id: str
    alpha: float = 0.1
    uq_methods: tuple[str, ...] = ("split",)
    skip_cached: bool = True
    n_samples: int = 0
    temperature_sampling: float = 1.0
    strategy: Literal["all", "single", "escalation", "bandit"] = "all"

    @property
    def needs_sampling(self) -> bool:
        """True iff any selected UQ method requires the sampling pass."""
        return any(m in _SAMPLING_UQ_METHODS for m in self.uq_methods)

    def effective_n_samples(self) -> int:
        """Return `n_samples` if set, else a sensible default when sampling is needed.

        Farquhar et al. (2024) use 10 samples; that is our default when
        sampling UQ methods are requested but the user didn't set
        `n_samples` explicitly.
        """
        if self.n_samples > 0:
            return self.n_samples
        if self.needs_sampling:
            return 10
        return 0


@dataclass(slots=True)
class JudgeRef:
    """Pairs a judge with the prompt template hash and routing metadata.

    `cost_tier` is read by the router to pick cheap-first / escalation
    schedules; defaults to 'mid' which is fine for AllJudges / SingleJudge.
    """

    judge: Judge
    prompt_version_hash: str
    cost_tier: Literal["cheap", "mid", "expensive"] = "mid"


async def run_evaluation(
    *,
    items: Sequence[BenchmarkItem],
    responses: dict[str, str],
    judges: Sequence[JudgeRef],
    store: DuckDBStore,
    cache: PromptCache | None,
    config: EvalConfig,
    model_under_test: str,
    nli_backend: NLIBackend | None = None,
    router: JuryRouter | None = None,
) -> CostReport:
    """Run point pass and (optionally) sampling pass. Returns per-judge cost.

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
            "n_samples": config.effective_n_samples(),
            "temperature_sampling": config.temperature_sampling,
            "strategy": config.strategy,
            "model_under_test": model_under_test,
        },
    )
    cost = _CostAccumulator()
    n_samples = config.effective_n_samples()
    if router is None:
        router = AllJudges()
    catalog = JudgeCatalog(
        judges=[JudgeMeta(judge_id=ref.judge.judge_id, cost_tier=ref.cost_tier) for ref in judges]
    )
    judge_by_id: dict[str, JudgeRef] = {ref.judge.judge_id: ref for ref in judges}

    if "semantic-entropy" in config.uq_methods and nli_backend is None:
        raise ValueError(
            "uq=semantic-entropy requires an `nli_backend` argument; pass an "
            "LLMNLIBackend or DebertaNLIBackend to run_evaluation()."
        )

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

        # Router: which judges to call first?
        initial_decision = router.initial(item, catalog)
        initial_refs = [judge_by_id[jid] for jid in initial_decision.judge_ids if jid in judge_by_id]
        point_responses = await _point_pass(item, response_text, initial_refs, cache, config)
        for jr in point_responses:
            record.judge_responses.append(jr)
            cost.add(jr)

        # Router: any escalation after seeing the initial responses?
        escalation_decision = router.escalate(item, catalog, point_responses)
        if escalation_decision.judge_ids:
            log.debug("escalation: %s", escalation_decision.reason)
            escalation_refs = [
                judge_by_id[jid]
                for jid in escalation_decision.judge_ids
                if jid in judge_by_id
            ]
            extra = await _point_pass(item, response_text, escalation_refs, cache, config)
            for jr in extra:
                record.judge_responses.append(jr)
                cost.add(jr)
                point_responses.append(jr)

        # Sampling pass for sampling-based UQ methods. Uses the same set
        # of judges as the (post-escalation) point pass.
        sampling_refs = [judge_by_id[r.judge_id] for r in point_responses if r.judge_id in judge_by_id]
        # Deduplicate while preserving order (a judge may appear once in
        # initial and once in escalation).
        seen: set[str] = set()
        deduped_sampling_refs: list[JudgeRef] = []
        for ref in sampling_refs:
            if ref.judge.judge_id in seen:
                continue
            seen.add(ref.judge.judge_id)
            deduped_sampling_refs.append(ref)

        samples_by_judge: dict[str, list[JudgeResponse]] = {}
        if n_samples > 0 and deduped_sampling_refs:
            samples_by_judge = await _sampling_pass(
                item, response_text, deduped_sampling_refs, config, n_samples
            )
            for jr_list in samples_by_judge.values():
                for jr in jr_list:
                    cost.add(jr)
            # Persist sample rows alongside point-pass rows.
            for jr_list in samples_by_judge.values():
                store.write_rows(row_from_response(record, jr) for jr in jr_list)

        # Aggregation + conformal.
        if record.judge_responses:
            decision = _aggregate_point(record.judge_responses, strategy=config.strategy)
            if "split" in config.uq_methods:
                conformal = _fit_split_for_decision(record.judge_responses, alpha=config.alpha)
                if conformal is not None:
                    decision.conformal.append(conformal)
            # Aleatoric/epistemic decomposition when sampling pass produced
            # ≥2 judges with ≥2 samples each.
            if "decomposition" in config.uq_methods and samples_by_judge:
                arrays = {
                    judge_id: np.asarray([s.score.value for s in sample_list], dtype=np.float64)
                    for judge_id, sample_list in samples_by_judge.items()
                    if len(sample_list) >= 2
                }
                if len(arrays) >= 2:
                    decomp = decompose_variance(arrays, alpha=config.alpha)
                    decision.decomposition = UncertaintyDecomposition(
                        total=decomp.total,
                        aleatoric=decomp.aleatoric,
                        epistemic=decomp.epistemic,
                        n_judges=decomp.n_judges,
                        n_samples_per_judge=decomp.n_samples_per_judge[0],
                    )
                    store.write_uq_result(
                        run_id=config.run_id,
                        item_id=item.item_id,
                        judge_id="__aggregate__",
                        method="decomposition",
                        value={
                            "total": decomp.total,
                            "aleatoric": decomp.aleatoric,
                            "epistemic": decomp.epistemic,
                            "aleatoric_ci": [decomp.aleatoric_ci_low, decomp.aleatoric_ci_high],
                            "epistemic_ci": [decomp.epistemic_ci_low, decomp.epistemic_ci_high],
                            "n_judges": decomp.n_judges,
                            "n_samples_per_judge": list(decomp.n_samples_per_judge),
                        },
                    )
            record.jury_decision = decision

        # Router online update after seeing the full set of responses.
        router.update(item, catalog, point_responses)

        # Sampling-based UQ metrics, written to judge_uq_results.
        if samples_by_judge and "self-consistency" in config.uq_methods:
            for judge_id, sample_list in samples_by_judge.items():
                values = np.asarray([s.score.value for s in sample_list], dtype=np.float64)
                if values.shape[0] >= 2:
                    sc = self_consistency_stats(values, alpha=config.alpha)
                    store.write_uq_result(
                        run_id=config.run_id,
                        item_id=item.item_id,
                        judge_id=judge_id,
                        method="self-consistency",
                        value={
                            "mean": sc.mean,
                            "variance": sc.variance,
                            "iqr": sc.iqr,
                            "ci_low": sc.ci_low,
                            "ci_high": sc.ci_high,
                            "alpha": sc.alpha,
                            "n_samples": sc.n_samples,
                        },
                    )
        if samples_by_judge and "semantic-entropy" in config.uq_methods:
            assert nli_backend is not None  # validated above
            for judge_id, sample_list in samples_by_judge.items():
                # Use rationale strings as the semantic content to cluster, falling
                # back to raw_text when rationale is empty. The judge's *score* is
                # numeric; semantic entropy is over the natural-language justification.
                texts = [s.score.rationale or s.raw_text for s in sample_list]
                if len(texts) >= 2:
                    se = await semantic_entropy(texts, nli=nli_backend)
                    store.write_uq_result(
                        run_id=config.run_id,
                        item_id=item.item_id,
                        judge_id=judge_id,
                        method="semantic-entropy",
                        value={
                            "entropy": se.entropy,
                            "n_clusters": se.n_clusters,
                            "cluster_sizes": list(se.cluster_sizes),
                            "n_samples": len(sample_list),
                        },
                    )

        store.write_rows(row_from_response(record, jr) for jr in point_responses)

    log.info(
        "Pipeline finished: %d items, %d judge calls (point + sample)",
        len(items),
        cost.n_calls,
    )
    return cost.to_report()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _point_pass(
    item: BenchmarkItem,
    response_text: str,
    judges: Sequence[JudgeRef],
    cache: PromptCache | None,
    config: EvalConfig,
) -> list[JudgeResponse]:
    """Fan out one temp=0 call per judge; honor prompt cache."""

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


async def _sampling_pass(
    item: BenchmarkItem,
    response_text: str,
    judges: Sequence[JudgeRef],
    config: EvalConfig,
    n_samples: int,
) -> dict[str, list[JudgeResponse]]:
    """Fan out `n_samples` temp>0 calls per judge for this item.

    Sampling responses bypass the prompt cache: each sample is an independent
    draw and re-sampling on rerun is the correct behavior.
    """

    async def _one(ref: JudgeRef, sample_idx: int) -> JudgeResponse:
        return await ref.judge.evaluate(
            item,
            response_text,
            sample_index=sample_idx + 1,  # reserve 0 for the point pass
            temperature=config.temperature_sampling,
        )

    tasks: list[asyncio.Task[JudgeResponse]] = []
    judge_for_task: list[str] = []
    for ref in judges:
        for k in range(n_samples):
            tasks.append(asyncio.create_task(_one(ref, k)))
            judge_for_task.append(ref.judge.judge_id)
    results = await asyncio.gather(*tasks)
    by_judge: dict[str, list[JudgeResponse]] = {}
    for judge_id, r in zip(judge_for_task, results, strict=True):
        by_judge.setdefault(judge_id, []).append(r)
    return by_judge


def _aggregate_point(
    responses: list[JudgeResponse],
    *,
    strategy: Literal["all", "single", "escalation", "bandit"] = "all",
) -> JuryDecision:
    """Simple mean aggregator. For routing strategies with ≥ 2 judges that
    want the full posterior, see `uq/disagreement.HierarchicalGaussianAggregator`.
    The mean is correct for single-judge / single-call routing and is a
    sensible fallback when EM would be ill-conditioned.
    """
    values = np.asarray([r.score.value for r in responses], dtype=np.float64)
    mean = float(values.mean())
    var = float(values.var(ddof=1)) if len(values) > 1 else 0.0
    return JuryDecision(
        item_id=responses[0].item_id,
        posterior_mean=mean,
        posterior_var=var,
        judges_called=[r.judge_id for r in responses],
        cost_usd=sum(r.cost_usd for r in responses),
        strategy=strategy,
    )


def _fit_split_for_decision(
    responses: list[JudgeResponse], *, alpha: float
) -> ConformalResult | None:
    """Build a split-conformal interval around the mean using inter-judge spread.

    This is a stand-in calibration when ground-truth labels are not yet
    available: residuals are |score_j - mean|. With the calibration probe
    in `benchmarks/calibration_probe.py`, this is replaced by proper
    held-out calibration.
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
