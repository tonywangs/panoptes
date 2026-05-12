"""Bootstrap routines used across PANOPTES's diagnostics modules.

All routines accept an explicit `rng` (`numpy.random.Generator`) so results
are reproducible. The `n_bootstrap` defaults of 1000–2000 are the standard
"good enough" range for most reliability / coverage applications; bump to
≥ 5000 when reporting in a paper.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CI:
    """Confidence interval (point + low/high)."""

    point: float
    low: float
    high: float


def pivot_ci(
    values: NDArray[np.floating],
    point: float,
    *,
    alpha: float = 0.1,
) -> CI:
    """Pivot (basic) bootstrap CI from a bootstrap distribution `values` and
    a *plug-in* point estimate `point`. See Efron & Tibshirani (1993) §13.5.

    Form: `[2θ̂ - θ̂*_{1-α/2}, 2θ̂ - θ̂*_{α/2}]` where `θ̂*` are bootstrap
    replicates. Pivot is preferred over percentile when the bootstrap
    distribution is approximately translation-symmetric around `point`.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D; got shape {values.shape}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    lo_q = float(np.quantile(values, 1.0 - alpha / 2.0))
    hi_q = float(np.quantile(values, alpha / 2.0))
    return CI(point=point, low=2.0 * point - lo_q, high=2.0 * point - hi_q)


def paired_bootstrap_diff[T: float | NDArray[np.floating]](
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    statistic: Callable[[NDArray[np.floating], NDArray[np.floating]], T],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[T, NDArray[np.floating]]:
    """Paired-difference bootstrap: resample indices, compute `statistic(a, b)`.

    Returns the point estimate and the bootstrap replicate distribution.
    Caller decides which CI form to use (`pivot_ci`, percentile, BCa).
    Paired resampling preserves the `(a_i, b_i)` correlation structure —
    essential when comparing two judges scoring the same items.
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape; got {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"a, b must be 1-D; got {a.shape}")
    del alpha  # passed for API parity with pivot_ci; we return the replicates
    rand = rng if rng is not None else np.random.default_rng()
    point = statistic(a, b)
    n = a.shape[0]
    replicates = np.empty(n_bootstrap, dtype=np.float64)
    for k in range(n_bootstrap):
        idx = rand.integers(0, n, size=n)
        rep = statistic(a[idx], b[idx])
        replicates[k] = float(rep)  # type: ignore[arg-type]
    return point, replicates


def bayesian_bootstrap_mean(
    samples: NDArray[np.floating],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.1,
    rng: np.random.Generator | None = None,
) -> CI:
    """Bayesian bootstrap (Rubin 1981) for the mean: Dirichlet(1,...,1) weights.

    Avoids ties from discrete resampling and corresponds to the posterior of
    a noninformative Dirichlet-process prior. Returns a percentile CI on
    the weighted mean.
    """
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1-D; got shape {samples.shape}")
    n = samples.shape[0]
    if n < 2:
        raise ValueError(f"need ≥ 2 samples; got n={n}")
    rand = rng if rng is not None else np.random.default_rng()
    raw = rand.standard_gamma(shape=1.0, size=(n_bootstrap, n))
    weights = raw / raw.sum(axis=1, keepdims=True)
    means = weights @ samples
    return CI(
        point=float(samples.mean()),
        low=float(np.quantile(means, alpha / 2.0)),
        high=float(np.quantile(means, 1.0 - alpha / 2.0)),
    )
