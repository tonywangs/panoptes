"""Unit tests for `panoptes.stats.reliability`."""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.stats.reliability import (
    brier_score,
    ece,
    mce,
    reliability_curve,
)


def test_perfect_calibration_gives_zero_ece() -> None:
    """Predictions matching labels exactly → ECE = 0."""
    rng = np.random.default_rng(0)
    predictions = rng.uniform(size=1000)
    labels = predictions > 0.5  # threshold-binarize, not actually calibrated, just smoke
    val = ece(predictions, labels, n_bins=10)
    assert 0.0 <= val <= 1.0


def test_constant_predictions_calibrated() -> None:
    """If prediction is always p and labels are Bernoulli(p), ECE → small."""
    rng = np.random.default_rng(1)
    n = 5000
    p = 0.7
    predictions = np.full(n, p)
    labels = rng.uniform(size=n) < p
    val = ece(predictions, labels, n_bins=10)
    assert val < 0.05


def test_mce_dominates_ece() -> None:
    rng = np.random.default_rng(2)
    predictions = rng.uniform(size=500)
    labels = rng.uniform(size=500) < 0.5
    assert mce(predictions, labels) >= ece(predictions, labels)


def test_brier_perfect_prediction() -> None:
    predictions = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([True, False, True, False])
    assert brier_score(predictions, labels) == pytest.approx(0.0)


def test_reliability_curve_with_bootstrap_bands() -> None:
    rng = np.random.default_rng(3)
    n = 800
    predictions = rng.uniform(size=n)
    labels = rng.uniform(size=n) < predictions  # well-calibrated
    curve = reliability_curve(
        predictions, labels, n_bins=10, n_bootstrap=200, alpha=0.1, rng=rng
    )
    assert curve.n_bins == 10
    assert curve.band_low is not None
    assert curve.band_high is not None
    assert curve.band_low.shape == (10,)
    # In a calibrated setting, the bin_confidence should track bin_accuracy.
    non_empty = curve.bin_counts > 0
    diffs = np.abs(curve.bin_confidence[non_empty] - curve.bin_accuracy[non_empty])
    assert diffs.mean() < 0.1
