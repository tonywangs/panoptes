"""Aleatoric/epistemic variance decomposition for jury verdicts.

For a `(judge, item)` system where each judge returns multiple temperature
samples, total predictive variance decomposes by the law of total variance:

    Var[score] = E_judge[Var(score | judge)]  +  Var_judge[E(score | judge)]
                 └─ aleatoric (irreducible) ─┘    └─── epistemic (reducible) ───┘

Intuition:
    - **aleatoric** is what's left when you fix the judge — the sampling
      noise *inside* that judge's response distribution. You can't reduce
      it by calling more judges.
    - **epistemic** is the variation *between* judges' expected scores.
      You can reduce it by calling more judges or by escalating to a
      higher-precision one.

Estimator: **nested resampling**. We have a list of judge IDs J and for each
judge `n_samples` MC draws (the sampling pass in `pipeline.py`).

1. Inner: per-judge sample mean `m_j` and sample variance `v_j`.
2. Aleatoric = mean over judges of `v_j` (weighted by `n_j` if uneven).
3. Epistemic = sample variance of `{m_j}`.
4. Total = aleatoric + epistemic.
5. Bootstrap CIs come from resampling the per-judge sample lists.

Per Kendall & Gal (2017) and Depeweg et al. (2018). This is the "model
disagreement" framing applied to LLM judges instead of NN ensembles.

References
----------
- Kendall, Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS.
- Depeweg, Hernández-Lobato, Doshi-Velez, Udluft (2018). *Decomposition of Uncertainty in Bayesian Deep Learning.* ICML.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Decomposition:
    """Variance decomposition + bootstrap CIs."""

    total: float
    aleatoric: float
    epistemic: float
    aleatoric_ci_low: float
    aleatoric_ci_high: float
    epistemic_ci_low: float
    epistemic_ci_high: float
    n_judges: int
    n_samples_per_judge: tuple[int, ...]


def decompose_variance(
    samples_by_judge: dict[str, NDArray[np.float64]] | dict[str, NDArray[np.floating]],
    *,
    alpha: float = 0.1,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
) -> Decomposition:
    """Compute aleatoric/epistemic variance + bootstrap CIs.

    Parameters
    ----------
    samples_by_judge : dict[judge_id, array of MC samples]
        Each judge contributes a 1-D array of scores. At least 2 judges
        with ≥ 2 samples each are required.
    alpha : float
        Two-sided miscoverage rate for bootstrap CIs (defaults to 0.1).
    n_bootstrap : int
        Bootstrap replicates for the CIs.
    rng : numpy Generator, optional

    Returns
    -------
    Decomposition
    """
    if len(samples_by_judge) < 2:
        raise ValueError("variance decomposition needs ≥ 2 judges")
    rand = rng if rng is not None else np.random.default_rng()

    judge_ids = sorted(samples_by_judge)
    per_judge: list[NDArray[np.floating]] = [
        np.asarray(samples_by_judge[j], dtype=np.float64) for j in judge_ids
    ]
    sample_counts = tuple(int(arr.shape[0]) for arr in per_judge)
    if any(n < 2 for n in sample_counts):
        raise ValueError(
            f"each judge needs ≥ 2 samples; got counts {sample_counts}"
        )

    point = _decompose_arrays(per_judge)
    boot_aleatoric: list[float] = []
    boot_epistemic: list[float] = []
    for _ in range(n_bootstrap):
        # Outer: resample judges with replacement (epistemic).
        judge_pick = rand.integers(0, len(per_judge), size=len(per_judge))
        resampled: list[NDArray[np.floating]] = []
        for j_idx in judge_pick:
            arr = per_judge[j_idx]
            # Inner: resample samples within this judge (aleatoric).
            inner_pick = rand.integers(0, arr.shape[0], size=arr.shape[0])
            resampled.append(arr[inner_pick])
        d = _decompose_arrays(resampled)
        boot_aleatoric.append(d[0])
        boot_epistemic.append(d[1])

    a_lo = float(np.quantile(boot_aleatoric, alpha / 2.0))
    a_hi = float(np.quantile(boot_aleatoric, 1.0 - alpha / 2.0))
    e_lo = float(np.quantile(boot_epistemic, alpha / 2.0))
    e_hi = float(np.quantile(boot_epistemic, 1.0 - alpha / 2.0))
    aleatoric, epistemic = point
    return Decomposition(
        total=aleatoric + epistemic,
        aleatoric=aleatoric,
        epistemic=epistemic,
        aleatoric_ci_low=a_lo,
        aleatoric_ci_high=a_hi,
        epistemic_ci_low=e_lo,
        epistemic_ci_high=e_hi,
        n_judges=len(per_judge),
        n_samples_per_judge=sample_counts,
    )


def _decompose_arrays(per_judge: list[NDArray[np.floating]]) -> tuple[float, float]:
    """Single-shot decomposition of already-fixed per-judge sample lists.

    Returns
    -------
    (aleatoric, epistemic) : tuple of floats
    """
    means: list[float] = []
    variances: list[float] = []
    weights: list[float] = []  # n_j for weighted aleatoric average
    for arr in per_judge:
        means.append(float(arr.mean()))
        variances.append(float(arr.var(ddof=1)) if arr.shape[0] > 1 else 0.0)
        weights.append(float(arr.shape[0]))
    weights_arr = np.asarray(weights)
    variances_arr = np.asarray(variances)
    means_arr = np.asarray(means)
    # Aleatoric: pooled (sample-weighted) within-judge variance.
    aleatoric = float((variances_arr * weights_arr).sum() / weights_arr.sum())
    # Epistemic: sample variance of judge-conditional means.
    epistemic = float(means_arr.var(ddof=1)) if len(means_arr) > 1 else 0.0
    return (aleatoric, epistemic)
