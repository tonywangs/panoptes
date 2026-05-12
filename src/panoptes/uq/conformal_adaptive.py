"""Conformalized Quantile Regression (CQR) for input-adaptive intervals.

CQR (Romano, Patterson, Candès 2019) extends split conformal so the interval
width *depends on the input*. The recipe is:

    1. **Train**: on a training split, fit two quantile regressors
       `q̂_low ≈ Q_{α/2}(Y | X)` and `q̂_high ≈ Q_{1-α/2}(Y | X)`.
    2. **Calibrate**: on a held-out calibration split, compute the
       *signed* conformity score
            E_i = max(q̂_low(X_i) - Y_i, Y_i - q̂_high(X_i))
       which is positive when `Y_i` falls outside `[q̂_low(X_i), q̂_high(X_i)]`
       and negative otherwise.
    3. **Predict**: let `Q_E` be the `ceil((n+1)(1-α))/n` quantile of
       `{E_1, ..., E_n}`. The interval is
            C(x) = [q̂_low(x) - Q_E, q̂_high(x) + Q_E].

The marginal coverage guarantee `P(Y ∈ C(X)) >= 1 - α` is the same as split
conformal — CQR trades the constant-width interval for one that *shrinks*
where the quantile regressors are confident and *expands* where they aren't.
"Adaptive" means input-adaptive, not online-adaptive.

For PANOPTES we use `sklearn.ensemble.GradientBoostingRegressor(loss='quantile')`
as the quantile-regression engine; it is non-parametric, handles low-dim
mixed-type features, and stays predictable on few hundred training points.
The feature builder lives in this module so callers can pass `JudgeResponse`
objects directly without reshaping.

References
----------
- Romano, Patterson, Candès (2019). *Conformalized Quantile Regression.* NeurIPS.
- Angelopoulos, Bates (2023). *A Gentle Introduction to Conformal Prediction
  and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor


def _conformal_quantile(scores: NDArray[np.floating], alpha: float) -> float:
    """The `ceil((n+1)(1-alpha))/n` empirical quantile, returning `+inf`
    when no finite rank achieves the requested coverage."""
    n = scores.shape[0]
    if n == 0:
        raise ValueError("conformity scores must be nonempty")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf
    return float(np.sort(scores)[rank - 1])


@dataclass(slots=True)
class AdaptiveConformal:
    """Fitted CQR predictor.

    Construct via `AdaptiveConformal.fit(X_train, y_train, X_cal, y_cal, alpha=...)`.
    `alpha` is fixed at training time because the quantile-regression
    estimators are α-specific (they predict the α/2 and 1-α/2 quantiles).
    To sweep coverage, re-fit per `alpha`.
    """

    q_low: GradientBoostingRegressor
    q_high: GradientBoostingRegressor
    conformity_scores: NDArray[np.floating]
    alpha: float
    score_lo: float
    score_hi: float

    @classmethod
    def fit(
        cls,
        x_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        x_cal: NDArray[np.floating],
        y_cal: NDArray[np.floating],
        *,
        alpha: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int | None = 0,
        score_lo: float = 0.0,
        score_hi: float = 1.0,
    ) -> AdaptiveConformal:
        """Train quantile regressors then compute calibration conformity scores."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        x_cal = np.asarray(x_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64)
        if x_train.ndim != 2:
            raise ValueError(f"x_train must be 2-D (n, d); got shape {x_train.shape}")
        if x_cal.ndim != 2:
            raise ValueError(f"x_cal must be 2-D (n, d); got shape {x_cal.shape}")
        if y_train.shape != (x_train.shape[0],):
            raise ValueError(
                f"y_train shape {y_train.shape} incompatible with x_train {x_train.shape}"
            )
        if y_cal.shape != (x_cal.shape[0],):
            raise ValueError(
                f"y_cal shape {y_cal.shape} incompatible with x_cal {x_cal.shape}"
            )

        q_low = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha / 2.0,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        q_high = GradientBoostingRegressor(
            loss="quantile",
            alpha=1.0 - alpha / 2.0,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        q_low.fit(x_train, y_train)
        q_high.fit(x_train, y_train)

        lo_cal = np.asarray(q_low.predict(x_cal), dtype=np.float64)
        hi_cal = np.asarray(q_high.predict(x_cal), dtype=np.float64)
        scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        return cls(
            q_low=q_low,
            q_high=q_high,
            conformity_scores=scores,
            alpha=alpha,
            score_lo=score_lo,
            score_hi=score_hi,
        )

    def predict_interval(
        self,
        x: NDArray[np.floating],
        *,
        alpha: float | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return `(lo, hi)` arrays of shape `(n,)`, clipped to score bounds.

        `alpha` defaults to the value supplied at `fit`. Passing a different
        `alpha` is allowed but discouraged: the quantile regressors were
        trained for the original `alpha`, so off-train α inflates / collapses
        the band asymmetrically. Use `fit` again for a different target rate.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (n, d); got shape {x.shape}")
        effective_alpha = self.alpha if alpha is None else alpha
        q = _conformal_quantile(self.conformity_scores, effective_alpha)
        lo_pred = np.asarray(self.q_low.predict(x), dtype=np.float64)
        hi_pred = np.asarray(self.q_high.predict(x), dtype=np.float64)
        if math.isinf(q):
            lo = np.full_like(lo_pred, self.score_lo)
            hi = np.full_like(hi_pred, self.score_hi)
            return (lo, hi)
        lo = np.clip(lo_pred - q, self.score_lo, self.score_hi)
        hi = np.clip(hi_pred + q, self.score_lo, self.score_hi)
        # Guarantee lo <= hi after clipping (numerical edge case).
        hi = np.maximum(hi, lo)
        return (lo, hi)

    def quantile(self, alpha: float | None = None) -> float:
        """Conformity-score quantile `Q_E` at the requested miscoverage."""
        effective_alpha = self.alpha if alpha is None else alpha
        return _conformal_quantile(self.conformity_scores, effective_alpha)
