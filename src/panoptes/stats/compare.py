"""Pairwise judge comparison.

Two questions PANOPTES needs to answer:

1. **Are judges A and B ranking items similarly?** — paired bootstrap on
   Spearman / Kendall rank correlation gives a CI on the correlation
   between their per-item scores.
2. **Do A and B disagree more than chance?** — permutation test on the
   `|score_A - score_B|` distribution against the null that the labels
   A / B are exchangeable for each item.

Paired resampling (not independent) is essential: the two judges score
*the same items*, and the per-item correlation between their scores is
the structure we want to preserve.

References
----------
- Pitman (1937). *Significance tests which may be applied to samples from any populations.* JRSS.
- Davison, Hinkley (1997). *Bootstrap Methods and their Application.*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from panoptes.stats.bootstrap import paired_bootstrap_diff


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    """Rank correlation + paired-bootstrap CI."""

    point: float
    ci_low: float
    ci_high: float
    n: int
    alpha: float
    method: str  # "spearman" | "kendall"


@dataclass(frozen=True, slots=True)
class PermutationResult:
    """Permutation test for A-vs-B label exchangeability."""

    observed: float
    p_value: float
    n_permutations: int


def paired_bootstrap_spearman(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.1,
    rng: np.random.Generator | None = None,
) -> CorrelationResult:
    """Spearman ρ with paired-bootstrap percentile CI."""

    def stat(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        res = stats.spearmanr(x, y)
        return float(res.statistic)  # type: ignore[attr-defined]

    point, replicates = paired_bootstrap_diff(
        a, b, stat, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng
    )
    return CorrelationResult(
        point=float(point),
        ci_low=float(np.quantile(replicates, alpha / 2.0)),
        ci_high=float(np.quantile(replicates, 1.0 - alpha / 2.0)),
        n=int(a.shape[0]),
        alpha=alpha,
        method="spearman",
    )


def paired_bootstrap_kendall(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.1,
    rng: np.random.Generator | None = None,
) -> CorrelationResult:
    """Kendall τ with paired-bootstrap percentile CI."""

    def stat(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        res = stats.kendalltau(x, y)
        return float(res.statistic)  # type: ignore[attr-defined]

    point, replicates = paired_bootstrap_diff(
        a, b, stat, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng
    )
    return CorrelationResult(
        point=float(point),
        ci_low=float(np.quantile(replicates, alpha / 2.0)),
        ci_high=float(np.quantile(replicates, 1.0 - alpha / 2.0)),
        n=int(a.shape[0]),
        alpha=alpha,
        method="kendall",
    )


def permutation_test_disagreement(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    *,
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
) -> PermutationResult:
    """Permutation test: is mean |a - b| larger than under random A/B labeling?

    Under H0 the per-item label "A" vs "B" is exchangeable: we can randomly
    swap `(a_i, b_i)` per item and the test statistic distribution should
    be unchanged. Reported p-value is the right-tailed probability of
    observing the actual mean absolute difference or larger.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"a {arr_a.shape} and b {arr_b.shape} must match")
    rand = rng if rng is not None else np.random.default_rng()
    observed = float(np.mean(np.abs(arr_a - arr_b)))
    count = 0
    n = arr_a.shape[0]
    for _ in range(n_permutations):
        swap = rand.integers(0, 2, size=n).astype(bool)
        perm_a = np.where(swap, arr_b, arr_a)
        perm_b = np.where(swap, arr_a, arr_b)
        test = float(np.mean(np.abs(perm_a - perm_b)))
        if test >= observed:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return PermutationResult(
        observed=observed, p_value=float(p), n_permutations=n_permutations
    )
