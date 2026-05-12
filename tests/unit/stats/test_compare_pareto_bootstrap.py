"""Unit tests for bootstrap / compare / pareto modules."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from panoptes.stats.bootstrap import (
    bayesian_bootstrap_mean,
    paired_bootstrap_diff,
    pivot_ci,
)
from panoptes.stats.compare import (
    paired_bootstrap_kendall,
    paired_bootstrap_spearman,
    permutation_test_disagreement,
)
from panoptes.stats.pareto import coverage_width_pareto


def test_pivot_ci_centered_at_point_for_symmetric_distribution() -> None:
    rng = np.random.default_rng(0)
    replicates = rng.normal(loc=0.5, scale=0.1, size=2000)
    ci = pivot_ci(replicates, point=0.5, alpha=0.1)
    assert abs((ci.low + ci.high) / 2 - 0.5) < 0.02


def test_bayesian_bootstrap_brackets_true_mean() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.5, scale=0.1, size=200)
    ci = bayesian_bootstrap_mean(samples, n_bootstrap=2000, alpha=0.1, rng=rng)
    assert ci.low <= 0.5 <= ci.high


def test_paired_bootstrap_diff_returns_replicates() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, size=100)
    b = rng.normal(loc=0.5, size=100)
    point, reps = paired_bootstrap_diff(
        a, b, lambda x, y: float(np.mean(y - x)), n_bootstrap=500, rng=rng
    )
    assert isinstance(point, float)
    assert reps.shape == (500,)
    # Bootstrap replicates should be centered near the *sample* mean of (b - a),
    # not the population mean (sampling noise = sqrt(2/100) ~ 0.14).
    sample_diff = float((b - a).mean())
    assert abs(reps.mean() - sample_diff) < 0.05


def test_spearman_perfect_correlation() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(size=50)
    b = a * 2 + 0.1  # monotone in a
    result = paired_bootstrap_spearman(a, b, n_bootstrap=200, rng=rng)
    assert result.point == pytest.approx(1.0, abs=1e-9)


def test_kendall_perfect_correlation() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(size=50)
    b = a * 3
    result = paired_bootstrap_kendall(a, b, n_bootstrap=200, rng=rng)
    assert result.point == pytest.approx(1.0, abs=1e-9)


def test_permutation_test_low_disagreement_yields_high_p() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(size=100)
    b = a + 0.01 * rng.standard_normal(100)
    result = permutation_test_disagreement(a, b, n_permutations=500, rng=rng)
    # Small disagreement: p should be reasonably large (not rejecting H0).
    assert result.p_value > 0.1


def test_coverage_width_pareto_monotone_in_one_minus_alpha() -> None:
    rng = np.random.default_rng(0)
    residuals = np.abs(rng.standard_normal(500)) * 0.1
    points = coverage_width_pareto(residuals)
    # Widths should be non-decreasing as 1-alpha grows.
    widths = [p.mean_width for p in sorted(points, key=lambda p: 1.0 - p.alpha)]
    for w_lo, w_hi in itertools.pairwise(widths):
        assert w_hi >= w_lo - 1e-12
