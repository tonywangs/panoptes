"""Coverage-width Pareto sweep.

For a conformal predictor calibrated at one specific α, you can ask "what
would coverage and mean interval width look like at *other* α?" by re-
quantiling the same calibration residuals (no need to re-fit). The Pareto
sweep is the resulting trade-off curve; you overlay the theoretical
`coverage = 1 - α` line to spot under/over-coverage.

This module is a thin wrapper that produces the data points; visualization
lives in the dashboard / report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One point on the coverage-width frontier."""

    alpha: float
    target_coverage: float
    empirical_coverage: float
    mean_width: float


def coverage_width_pareto(
    residuals: NDArray[np.floating],
    *,
    test_residuals: NDArray[np.floating] | None = None,
    alphas: NDArray[np.floating] | None = None,
    quantile_fn: Callable[[NDArray[np.floating], float], float] | None = None,
) -> list[ParetoPoint]:
    """Sweep α and return (target, empirical, mean_width) triples.

    Parameters
    ----------
    residuals : 1-D
        Calibration residuals used to derive the quantile per α.
    test_residuals : 1-D, optional
        Held-out residuals used to estimate empirical coverage. Defaults
        to `residuals` (training-set evaluation — useful for plotting the
        in-sample frontier).
    alphas : array-like, optional
        Sweep grid; defaults to `np.linspace(0.01, 0.5, 50)`.
    quantile_fn : callable, optional
        Maps `(residuals, alpha) -> q`. Defaults to the standard split-
        conformal quantile from `panoptes.uq.conformal_split`. Pass an
        alternative (e.g. an `AdaptiveConformal.quantile` closure) for
        other conformal variants.
    """
    if quantile_fn is None:
        from panoptes.uq.conformal_split import split_conformal_quantile  # noqa: PLC0415

        quantile_fn = split_conformal_quantile
    cal = np.asarray(residuals, dtype=np.float64)
    test = (
        np.asarray(test_residuals, dtype=np.float64)
        if test_residuals is not None
        else cal
    )
    if alphas is None:
        alphas = np.linspace(0.01, 0.5, 50)
    points: list[ParetoPoint] = []
    for alpha in alphas:
        a = float(alpha)
        q = quantile_fn(cal, a)
        if not np.isfinite(q):
            continue
        emp_cov = float((test <= q).mean())
        points.append(
            ParetoPoint(
                alpha=a,
                target_coverage=1.0 - a,
                empirical_coverage=emp_cov,
                mean_width=2.0 * q,  # |y - point| ≤ q ⇒ width 2q
            )
        )
    return points
