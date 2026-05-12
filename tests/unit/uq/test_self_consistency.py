"""Unit tests for `panoptes.uq.self_consistency`."""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.uq.self_consistency import self_consistency_stats


def test_rejects_small_sample() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        self_consistency_stats(np.array([0.5]))


def test_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        self_consistency_stats(np.array([0.5, 0.7]), alpha=0.0)


def test_constant_samples_have_zero_dispersion() -> None:
    samples = np.array([0.6, 0.6, 0.6, 0.6])
    result = self_consistency_stats(samples)
    assert result.variance == pytest.approx(0.0)
    assert result.iqr == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.6)
    assert result.ci_high == pytest.approx(0.6)


def test_high_variance_widens_ci() -> None:
    rng = np.random.default_rng(seed=0)
    tight = 0.5 + 0.01 * rng.standard_normal(50)
    wide = 0.5 + 0.20 * rng.standard_normal(50)
    r_tight = self_consistency_stats(
        tight, alpha=0.1, n_bootstrap=2000, rng=np.random.default_rng(1)
    )
    r_wide = self_consistency_stats(
        wide, alpha=0.1, n_bootstrap=2000, rng=np.random.default_rng(1)
    )
    assert (r_wide.ci_high - r_wide.ci_low) > (r_tight.ci_high - r_tight.ci_low) * 3.0


def test_bootstrap_ci_brackets_mean_on_average() -> None:
    """With 90% target and centered Gaussian samples, CI should cover the mean."""
    rng = np.random.default_rng(seed=2)
    n_repeats = 200
    covered = 0
    true_mean = 0.5
    for i in range(n_repeats):
        local_rng = np.random.default_rng(seed=100 + i)
        samples = true_mean + 0.1 * local_rng.standard_normal(40)
        r = self_consistency_stats(
            samples, alpha=0.1, n_bootstrap=500, rng=rng
        )
        # Sample mean (not true mean) should be inside its own CI by construction.
        sample_mean = float(samples.mean())
        if r.ci_low <= sample_mean <= r.ci_high:
            covered += 1
    # The sample mean is always within its own bootstrap CI (>97% in practice).
    assert covered / n_repeats > 0.95
