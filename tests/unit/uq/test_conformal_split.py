"""Unit tests for `panoptes.uq.conformal_split`.

The headline test is a Monte-Carlo coverage check on a synthetic Gaussian
calibration set: with `n_cal` exchangeable calibration points and an
independently-drawn test set, the empirical coverage of the split-conformal
interval should track the nominal `1 - alpha` rate within the binomial
standard error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from panoptes.uq.conformal_split import SplitConformal, split_conformal_quantile


def test_quantile_rejects_bad_alpha() -> None:
    residuals = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="alpha"):
        split_conformal_quantile(residuals, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        split_conformal_quantile(residuals, alpha=1.0)


def test_quantile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        split_conformal_quantile(np.array([]), alpha=0.1)


def test_quantile_returns_inf_when_n_too_small() -> None:
    # With n=5, ceil((n+1) * (1-alpha)) > n when alpha < 1/6 ~= 0.167.
    # So alpha=0.05 should return +inf rather than silently undercover.
    residuals = np.linspace(0.0, 0.5, 5)
    q = split_conformal_quantile(residuals, alpha=0.05)
    assert math.isinf(q)


def test_quantile_is_a_known_order_statistic() -> None:
    # n=10, alpha=0.1 -> rank = ceil(11 * 0.9) = 10 -> 10th order stat = max.
    residuals = np.linspace(0.05, 0.95, 10)
    q = split_conformal_quantile(residuals, alpha=0.1)
    assert q == pytest.approx(0.95)


def test_predict_interval_clips_to_score_bounds() -> None:
    cp = SplitConformal(residuals=np.array([0.4, 0.4, 0.4]), score_lo=0.0, score_hi=1.0)
    lo, hi = cp.predict_interval(0.9, alpha=0.1)  # alpha small enough to return inf
    assert (lo, hi) == (0.0, 1.0)


def test_predict_interval_centered_when_q_finite() -> None:
    # 19 residuals, alpha=0.1 -> rank = ceil(20*0.9) = 18 -> 18th order stat.
    residuals = np.linspace(0.0, 0.18, 19)
    cp = SplitConformal(residuals=residuals)
    q = cp.quantile(alpha=0.1)
    lo, hi = cp.predict_interval(0.5, alpha=0.1)
    assert lo == pytest.approx(0.5 - q)
    assert hi == pytest.approx(0.5 + q)


def test_marginal_coverage_on_synthetic_gaussian() -> None:
    """Empirical 1-alpha coverage should hold on i.i.d. data with known noise.

    Setup: predictor mu_hat(x) is a constant 0.5; true score is
        y_i = 0.5 + sigma * Z_i, Z_i ~ N(0, 1).
    Then |y_i - mu_hat| = sigma * |Z_i| is half-normal. The empirical 1-alpha
    quantile of half-normal residuals should give a marginal coverage rate
    within ~2 standard errors of 1-alpha on a large held-out test set.
    """
    rng = np.random.default_rng(seed=1234)
    sigma = 0.1
    n_cal = 1000
    n_test = 5000
    alpha = 0.1

    cal_resid = sigma * np.abs(rng.standard_normal(n_cal))
    cp = SplitConformal(residuals=cal_resid, score_lo=-10.0, score_hi=10.0)
    q = cp.quantile(alpha=alpha)

    test_resid = sigma * np.abs(rng.standard_normal(n_test))
    coverage = float(np.mean(test_resid <= q))

    # Binomial SE at 90% on n=5000: sqrt(0.9*0.1/5000) ~ 0.0042. 4 SE ~ 0.017.
    assert coverage == pytest.approx(1.0 - alpha, abs=0.02)


def test_fit_validates_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        SplitConformal.fit(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="1-D"):
        SplitConformal.fit(np.array([[0.1], [0.2]]), np.array([[0.1], [0.2]]))
