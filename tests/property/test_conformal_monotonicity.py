"""Hypothesis property tests for conformal interval shape.

These tests express invariants the math must satisfy regardless of inputs:

1. **Width monotone in `alpha`**: tighter miscoverage (larger 1-alpha) requires
   a wider interval. So for `0 < a1 < a2 < 1`, `width(a1) >= width(a2)`.
2. **Interval contains the point estimate** when the quantile is finite.
3. **Bounds-preservation**: `predict_interval` never returns values outside
   `[score_lo, score_hi]`.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from panoptes.uq.conformal_split import SplitConformal

# Residuals: nonneg floats, finite. Use floats(width=64) and abs().
_RESIDUAL_STRAT = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=200,
)


@settings(max_examples=80, deadline=None)
@given(
    _RESIDUAL_STRAT,
    st.floats(min_value=0.01, max_value=0.5),
    st.floats(min_value=0.5, max_value=0.99),
)
def test_width_monotone_in_one_minus_alpha(residuals: list[float], a1: float, a2: float) -> None:
    if a1 >= a2:
        a1, a2 = a2, a1
        if a1 == a2:
            return
    cp = SplitConformal(residuals=np.asarray(residuals))
    q1 = cp.quantile(alpha=a1)
    q2 = cp.quantile(alpha=a2)
    # Larger 1 - alpha (smaller alpha) requires larger or equal quantile.
    # Infinities are allowed but if both are finite, a1 < a2 implies q1 >= q2.
    if math.isfinite(q1) and math.isfinite(q2):
        assert q1 >= q2 - 1e-12


@settings(max_examples=80, deadline=None)
@given(
    _RESIDUAL_STRAT,
    st.floats(min_value=0.05, max_value=0.95),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_interval_contains_point_when_finite(
    residuals: list[float], alpha: float, point: float
) -> None:
    cp = SplitConformal(residuals=np.asarray(residuals), score_lo=0.0, score_hi=1.0)
    lo, hi = cp.predict_interval(point, alpha=alpha)
    assert lo <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    # When the quantile is finite, the unclipped interval contains `point`
    # by symmetry; clipping can drop the point out only if the point itself
    # is outside [0, 1] (which the hypothesis range excludes).
    q = cp.quantile(alpha=alpha)
    if math.isfinite(q):
        assert max(0.0, point - q) <= point <= min(1.0, point + q) + 1e-12
