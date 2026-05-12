"""Coverage diagnostics for conformal-prediction outputs.

Three families:

1. **Marginal coverage** — what fraction of test points fell inside the
   prediction interval? Reported with Clopper-Pearson (exact binomial) CI.
2. **Conditional coverage** per task family — each group tested
   independently with Bonferroni-corrected p-values for the null
   hypothesis "this group's coverage equals 1 - α".
3. **Hosmer-Lemeshow binning test** — adapted for coverage: bin predictions
   by predicted score, compare observed-vs-expected coverage rates, χ² statistic.

The math is standard but PANOPTES needs all of it on every run.

References
----------
- Clopper, Pearson (1934). *The use of confidence or fiducial limits illustrated in the case of the binomial.* Biometrika.
- Hosmer, Lemeshow (1980). *Goodness-of-fit tests for the multiple logistic regression model.*
- Vovk (2012). *Conditional Validity of Inductive Conformal Predictors.*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """Marginal coverage + binomial CI."""

    coverage: float
    n: int
    n_covered: int
    ci_low: float
    ci_high: float
    target: float


@dataclass(frozen=True, slots=True)
class ConditionalCoverageResult:
    """Group-conditional coverage with Bonferroni-corrected p-values."""

    by_group: dict[str, CoverageResult]
    p_values: dict[str, float]
    bonferroni_p_values: dict[str, float]
    target: float
    n_groups: int


@dataclass(frozen=True, slots=True)
class HosmerLemeshowResult:
    """Hosmer-Lemeshow style binning test for coverage rate."""

    chi_squared: float
    df: int
    p_value: float
    n_bins: int


def clopper_pearson_ci(n_covered: int, n: int, *, alpha: float = 0.1) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial CI at level `1 - alpha`.

    Uses the Beta-distribution form: lower = Beta(α/2; k, n-k+1) and
    upper = Beta(1-α/2; k+1, n-k). Edge cases at k=0 or k=n are handled
    by setting the absent bound to 0 or 1 respectively.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if not 0 <= n_covered <= n:
        raise ValueError(f"n_covered ({n_covered}) must be in [0, n] ({n})")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n_covered == 0:
        low = 0.0
    else:
        low = float(stats.beta.ppf(alpha / 2.0, n_covered, n - n_covered + 1))
    if n_covered == n:
        high = 1.0
    else:
        high = float(stats.beta.ppf(1.0 - alpha / 2.0, n_covered + 1, n - n_covered))
    return (low, high)


def marginal_coverage(
    covered: NDArray[np.bool_],
    *,
    target: float,
    alpha: float = 0.1,
) -> CoverageResult:
    """Empirical coverage + Clopper-Pearson CI at level 1-alpha.

    `target` is the nominal coverage rate (1 - α_conformal). The returned
    `ci_low`, `ci_high` should bracket `target` if the conformal predictor
    is calibrated.
    """
    arr = np.asarray(covered).astype(bool)
    if arr.ndim != 1:
        raise ValueError(f"covered must be 1-D; got {arr.shape}")
    n = int(arr.shape[0])
    n_cov = int(arr.sum())
    coverage = n_cov / max(n, 1)
    ci_low, ci_high = clopper_pearson_ci(n_cov, n, alpha=alpha)
    return CoverageResult(
        coverage=coverage,
        n=n,
        n_covered=n_cov,
        ci_low=ci_low,
        ci_high=ci_high,
        target=target,
    )


def _binomial_two_sided_p(n_covered: int, n: int, *, target: float) -> float:
    """Two-sided exact binomial p-value for H0: P(cover) = target."""
    if n == 0:
        return 1.0
    p_obs = stats.binom.pmf(n_covered, n, target)
    pmfs = stats.binom.pmf(np.arange(n + 1), n, target)
    p_value = float(pmfs[pmfs <= p_obs + 1e-12].sum())
    return min(1.0, p_value)


def conditional_coverage_test(
    covered_by_group: dict[str, NDArray[np.bool_]],
    *,
    target: float,
    alpha: float = 0.1,
) -> ConditionalCoverageResult:
    """Per-group coverage + Bonferroni-corrected two-sided binomial p-values.

    Tests H0: `coverage_g = target` for each group g; Bonferroni adjusts for
    the number of simultaneous tests so the family-wise error rate stays
    ≤ alpha.
    """
    if not covered_by_group:
        raise ValueError("covered_by_group is empty")
    by_group: dict[str, CoverageResult] = {}
    raw_p: dict[str, float] = {}
    for group, covered in covered_by_group.items():
        result = marginal_coverage(covered, target=target, alpha=alpha)
        by_group[group] = result
        raw_p[group] = _binomial_two_sided_p(result.n_covered, result.n, target=target)
    k = len(covered_by_group)
    bonf_p = {g: min(1.0, p * k) for g, p in raw_p.items()}
    return ConditionalCoverageResult(
        by_group=by_group,
        p_values=raw_p,
        bonferroni_p_values=bonf_p,
        target=target,
        n_groups=k,
    )


def hosmer_lemeshow_test(
    predictions: NDArray[np.floating],
    covered: NDArray[np.bool_],
    *,
    target: float,
    n_bins: int = 10,
) -> HosmerLemeshowResult:
    """Hosmer-Lemeshow style χ² test on bin-level coverage rates.

    Predictions are sorted and split into `n_bins` equal-frequency bins; in
    each bin we compute observed covered count, expected count under
    H0: P(cover|bin) = target, and accumulate `(O - E)² / (E (1-E)/n_b)`.
    Asymptotically χ² with `n_bins - 2` df under the null.
    """
    arr_pred = np.asarray(predictions, dtype=np.float64)
    arr_cov = np.asarray(covered).astype(bool)
    if arr_pred.shape != arr_cov.shape:
        raise ValueError(
            f"predictions {arr_pred.shape} and covered {arr_cov.shape} must match"
        )
    n = int(arr_pred.shape[0])
    if n < n_bins:
        raise ValueError(f"n={n} too small for n_bins={n_bins}")
    order = np.argsort(arr_pred)
    bin_edges = np.array_split(order, n_bins)
    chi2 = 0.0
    for indices in bin_edges:
        if indices.size == 0:
            continue
        n_b = float(indices.size)
        observed = float(arr_cov[indices].sum())
        expected = target * n_b
        variance = max(target * (1.0 - target) * n_b, 1e-12)
        chi2 += (observed - expected) ** 2 / variance
    df = max(n_bins - 2, 1)
    p_value = float(1.0 - stats.chi2.cdf(chi2, df=df))
    return HosmerLemeshowResult(
        chi_squared=chi2,
        df=df,
        p_value=p_value,
        n_bins=n_bins,
    )
