"""Unit tests for `panoptes.uq.conformal_adaptive` (CQR).

Headline test: on heteroscedastic synthetic data, CQR intervals should be
*narrower in the low-noise region* and *wider in the high-noise region*.
A static split-conformal interval (constant width) cannot pass this test;
CQR's input-adaptive widths are the point of the method.
"""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.uq.conformal_adaptive import AdaptiveConformal


def _heteroscedastic_dataset(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """y | x ~ N(0.5, sigma(x)) with sigma small for x<0.5 and large for x>=0.5."""
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    sigma = np.where(x[:, 0] < 0.5, 0.02, 0.20)
    y = 0.5 + sigma * rng.standard_normal(n)
    return x, np.clip(y, 0.0, 1.0)


def test_fit_validates_shapes() -> None:
    x = np.zeros((5, 2))
    y_bad = np.zeros(4)
    with pytest.raises(ValueError, match="incompatible"):
        AdaptiveConformal.fit(x, y_bad, x, np.zeros(5))


def test_fit_validates_alpha() -> None:
    x = np.zeros((5, 1))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="alpha"):
        AdaptiveConformal.fit(x, y, x, y, alpha=1.5)


def test_intervals_within_bounds_and_ordered() -> None:
    rng = np.random.default_rng(seed=0)
    x_train, y_train = _heteroscedastic_dataset(200, rng)
    x_cal, y_cal = _heteroscedastic_dataset(200, rng)
    cqr = AdaptiveConformal.fit(x_train, y_train, x_cal, y_cal, alpha=0.1)
    x_test = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    lo, hi = cqr.predict_interval(x_test)
    assert np.all(lo >= 0.0)
    assert np.all(hi <= 1.0)
    assert np.all(lo <= hi + 1e-12)


def test_widths_narrower_in_low_noise_region() -> None:
    """Heteroscedasticity: low-noise region should get narrower intervals."""
    rng = np.random.default_rng(seed=42)
    x_train, y_train = _heteroscedastic_dataset(500, rng)
    x_cal, y_cal = _heteroscedastic_dataset(500, rng)
    cqr = AdaptiveConformal.fit(x_train, y_train, x_cal, y_cal, alpha=0.2)
    # Test points in both regimes.
    x_lo = np.linspace(0.05, 0.45, 20).reshape(-1, 1)
    x_hi = np.linspace(0.55, 0.95, 20).reshape(-1, 1)
    lo_low, hi_low = cqr.predict_interval(x_lo)
    lo_hi, hi_hi = cqr.predict_interval(x_hi)
    mean_width_low = float(np.mean(hi_low - lo_low))
    mean_width_hi = float(np.mean(hi_hi - lo_hi))
    # CQR is heteroscedasticity-aware: the high-noise region should be ≥1.5x
    # wider on average; using ≥1.2x to leave generous margin for the small
    # sample size. (True sigma ratio is 10x.)
    assert mean_width_hi > mean_width_low * 1.2
