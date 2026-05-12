"""Self-consistency variance for sampling-based uncertainty.

Given `n` temperature-sampled judge scores `(s_1, ..., s_n)` for the same
`(judge, item)` pair, compute:

    - Sample variance (unbiased, `ddof=1`)
    - Interquartile range (IQR = Q3 - Q1)
    - **Bayesian bootstrap CI** on the mean: draw `B` weight vectors
      `w^(b) ~ Dirichlet(1, ..., 1)` (the Bayesian bootstrap weights of
      Rubin 1981), form weighted means `μ^(b) = Σ w^(b)_i s_i`, and report
      the `(α/2, 1-α/2)` quantiles of `{μ^(b)}`.

The Bayesian bootstrap is preferred over the classical (Efron) bootstrap
here because it (a) avoids "ties" from the discrete resampling kernel —
problematic when `n` is small, and (b) corresponds to the posterior of the
nonparametric Bayesian model with a noninformative Dirichlet-process prior,
which is more honest in the small-`n` regime where we typically operate.

The "self-consistency" framing is from Wang et al. ICLR 2023 (chain-of-
thought stability under sampling). In PANOPTES, large MC variance is the
*sampling-aleatoric* component of the aleatoric/epistemic split (see
`uq/decomposition.py`).

References
----------
- Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou (2023).
  *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.
- Rubin (1981). *The Bayesian Bootstrap.* Annals of Statistics 9(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True, slots=True)
class SelfConsistencyResult:
    """Diagnostics over `n` temperature-sampled scores."""

    mean: float
    variance: float
    iqr: float
    ci_low: float
    ci_high: float
    alpha: float
    n_samples: int
    n_bootstrap: int


def self_consistency_stats(
    samples: NDArray[np.floating],
    *,
    alpha: float = 0.1,
    n_bootstrap: int = 2000,
    rng: np.random.Generator | None = None,
) -> SelfConsistencyResult:
    """Compute MC mean, variance, IQR, and Bayesian-bootstrap CI on the mean.

    Parameters
    ----------
    samples : 1-D array of judge scores
        The `n` temperature-sampled values for the same (judge, item) pair.
    alpha : float in (0, 1)
        Two-sided miscoverage rate for the bootstrap CI.
    n_bootstrap : int
        Number of Dirichlet weight draws.
    rng : numpy Generator, optional
        For reproducibility; defaults to a fresh `default_rng()`.

    Returns
    -------
    SelfConsistencyResult
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"samples must be 1-D; got shape {arr.shape}")
    n = arr.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 samples for self-consistency; got n={n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    rand = rng if rng is not None else np.random.default_rng()
    mean = float(arr.mean())
    variance = float(arr.var(ddof=1))
    iqr = float(stats.iqr(arr))

    # Bayesian bootstrap: w_b ~ Dirichlet(1, ..., 1) of length n.
    # `gamma(1, 1)` sample / sum normalizes to Dirichlet(1,..,1).
    raw_weights = rand.standard_gamma(shape=1.0, size=(n_bootstrap, n))
    weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
    boot_means = weights @ arr
    ci_low = float(np.quantile(boot_means, alpha / 2.0))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    return SelfConsistencyResult(
        mean=mean,
        variance=variance,
        iqr=iqr,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        n_samples=n,
        n_bootstrap=n_bootstrap,
    )
