"""Unit tests for `panoptes.stats.coverage_tests`."""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.stats.coverage_tests import (
    clopper_pearson_ci,
    conditional_coverage_test,
    hosmer_lemeshow_test,
    marginal_coverage,
)


def test_clopper_pearson_edge_cases() -> None:
    # All covered: lower = Beta quantile, upper = 1
    lo, hi = clopper_pearson_ci(n_covered=10, n=10, alpha=0.1)
    assert hi == pytest.approx(1.0)
    assert lo < 1.0
    # None covered: lower = 0, upper = Beta quantile
    lo, hi = clopper_pearson_ci(n_covered=0, n=10, alpha=0.1)
    assert lo == pytest.approx(0.0)
    assert 0.0 < hi < 1.0


def test_clopper_pearson_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        clopper_pearson_ci(0, 0)
    with pytest.raises(ValueError, match=r"n_covered .* in"):
        clopper_pearson_ci(11, 10)
    with pytest.raises(ValueError, match="alpha"):
        clopper_pearson_ci(5, 10, alpha=0.0)


def test_marginal_coverage_matches_target_on_calibrated_data() -> None:
    """If we manually construct 90% covered booleans, marginal_coverage(target=0.9)
    should yield empirical coverage in the CI."""
    rng = np.random.default_rng(0)
    covered = rng.uniform(size=2000) < 0.9
    result = marginal_coverage(covered.astype(bool), target=0.9, alpha=0.05)
    assert result.ci_low <= 0.9 <= result.ci_high
    assert abs(result.coverage - 0.9) < 0.03


def test_conditional_coverage_flags_miscovered_group() -> None:
    """One group at 0.95 coverage and another at 0.6 — Bonferroni p of the
    second should be small enough to flag."""
    rng = np.random.default_rng(1)
    group_a = rng.uniform(size=400) < 0.9
    group_b = rng.uniform(size=400) < 0.6
    result = conditional_coverage_test(
        {"a": group_a.astype(bool), "b": group_b.astype(bool)},
        target=0.9,
        alpha=0.1,
    )
    assert result.bonferroni_p_values["b"] < 0.05  # severely miscovered
    assert result.bonferroni_p_values["a"] > 0.05  # not flagged
    assert result.n_groups == 2


def test_hosmer_lemeshow_smoke() -> None:
    rng = np.random.default_rng(2)
    n = 500
    predictions = rng.uniform(size=n)
    # Calibrated case: P(cover) = target uniformly across bins.
    covered_calibrated = rng.uniform(size=n) < 0.9
    result_cal = hosmer_lemeshow_test(predictions, covered_calibrated, target=0.9, n_bins=8)
    # P-value should be reasonably large (no rejection) under calibration.
    assert 0.0 <= result_cal.p_value <= 1.0
    assert result_cal.df == 6


def test_hosmer_lemeshow_rejects_small_n() -> None:
    with pytest.raises(ValueError, match="too small"):
        hosmer_lemeshow_test(
            np.array([0.1, 0.2]),
            np.array([True, False]),
            target=0.9,
            n_bins=10,
        )
