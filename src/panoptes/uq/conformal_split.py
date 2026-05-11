"""Split conformal prediction for bounded-score regression.

Background
----------
Given exchangeable calibration pairs (X_i, y_i) for i in 1..n and a learned
point predictor mu_hat, split conformal forms a prediction interval

    C(x) = [mu_hat(x) - q, mu_hat(x) + q]

where `q` is the (ceil((n+1)(1-alpha)) / n)-th empirical quantile of the
calibration residuals |y_i - mu_hat(X_i)|. The +1 correction yields the
*finite-sample* marginal coverage guarantee

    P(y_{n+1} in C(X_{n+1})) >= 1 - alpha

with no parametric assumptions beyond exchangeability of the data.

References
----------
- Papadopoulos, Proedrou, Vovk, Gammerman (2002): *Inductive Confidence
  Machines for Regression*.
- Vovk, Gammerman, Shafer (2005): *Algorithmic Learning in a Random World*.
- Angelopoulos & Bates (2023): *A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification*. Tutorial.

PANOPTES-specific notes
-----------------------
- Scores live in [0, 1] (the rubric-judge value scale). After applying
  +-q to the point estimate we clip to [0, 1] because intervals leaving
  the bounded range are uninformative.
- Calibration residuals here are *symmetric absolute residuals*; for
  asymmetric / locally-adaptive variants see `conformal_adaptive.py` (M2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def split_conformal_quantile(residuals: NDArray[np.floating], alpha: float) -> float:
    """Return the conformal quantile `q` for the given residuals and miscoverage.

    Computes the (ceil((n+1)(1-alpha)) / n)-th empirical quantile, which is
    the finite-sample-valid threshold under exchangeability. When
    `ceil((n+1)(1-alpha)) > n` (i.e. alpha < 1/(n+1)), no finite quantile
    achieves the target coverage; we return `+inf` to signal "no useful
    bound" rather than silently undercovering.

    Parameters
    ----------
    residuals : array-like of nonneg floats
        Nonconformity scores from the held-out calibration split.
    alpha : float in (0, 1)
        Nominal miscoverage rate; target coverage is 1 - alpha.

    Returns
    -------
    float : conformal quantile `q`.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    arr = np.asarray(residuals, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"residuals must be 1-D; got shape {arr.shape}")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("residuals must be nonempty")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf
    # `rank` is a 1-based index; np.partition uses 0-based.
    sorted_arr = np.sort(arr)
    return float(sorted_arr[rank - 1])


@dataclass(slots=True)
class SplitConformal:
    """Fitted split-conformal predictor over bounded [0, 1] scores.

    Usage
    -----
    >>> from panoptes.uq import SplitConformal
    >>> import numpy as np
    >>> preds = np.array([0.7, 0.2, 0.9, 0.5])
    >>> labels = np.array([0.6, 0.3, 0.85, 0.55])
    >>> cp = SplitConformal.fit(preds, labels)
    >>> lo, hi = cp.predict_interval(0.8, alpha=0.1)
    >>> 0.0 <= lo <= hi <= 1.0
    True

    The interval is `[clip(point - q, 0, 1), clip(point + q, 0, 1)]` where
    `q` is the split-conformal quantile at the requested `alpha`. Marginal
    coverage is guaranteed by the underlying theorem under exchangeability.
    """

    residuals: NDArray[np.floating]
    score_lo: float = 0.0
    score_hi: float = 1.0

    @classmethod
    def fit(
        cls,
        predictions: NDArray[np.floating],
        labels: NDArray[np.floating],
        *,
        score_lo: float = 0.0,
        score_hi: float = 1.0,
    ) -> SplitConformal:
        """Compute calibration residuals from held-out (prediction, label) pairs."""
        preds = np.asarray(predictions, dtype=np.float64)
        labs = np.asarray(labels, dtype=np.float64)
        if preds.shape != labs.shape:
            raise ValueError(
                f"predictions {preds.shape} and labels {labs.shape} must have the same shape"
            )
        if preds.ndim != 1:
            raise ValueError(f"inputs must be 1-D; got shape {preds.shape}")
        residuals = np.abs(preds - labs)
        return cls(residuals=residuals, score_lo=score_lo, score_hi=score_hi)

    def quantile(self, alpha: float) -> float:
        """Return the conformal quantile `q` at miscoverage `alpha`."""
        return split_conformal_quantile(self.residuals, alpha)

    def predict_interval(self, point: float, *, alpha: float) -> tuple[float, float]:
        """Return `(lo, hi)` clipped to [score_lo, score_hi]."""
        q = self.quantile(alpha)
        if math.isinf(q):
            return (self.score_lo, self.score_hi)
        lo = max(self.score_lo, point - q)
        hi = min(self.score_hi, point + q)
        return (lo, hi)
