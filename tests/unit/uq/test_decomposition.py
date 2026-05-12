"""Unit tests for `panoptes.uq.decomposition.decompose_variance`."""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.uq.decomposition import decompose_variance


def test_pure_aleatoric_zero_epistemic() -> None:
    """All judges have the same mean → epistemic ≈ 0, aleatoric = inner var."""
    rng = np.random.default_rng(seed=0)
    samples = {
        f"j-{j}": rng.normal(0.5, 0.1, size=50) for j in range(4)
    }
    result = decompose_variance(samples, alpha=0.1, n_bootstrap=300, rng=rng)
    # All judges centered at 0.5; epistemic should be small relative to aleatoric.
    assert result.epistemic < result.aleatoric * 0.5
    assert result.aleatoric == pytest.approx(0.01, abs=0.005)


def test_pure_epistemic_zero_aleatoric() -> None:
    """Within-judge constant; between-judges varies → aleatoric ≈ 0."""
    rng = np.random.default_rng(seed=1)
    samples = {
        "j-0": np.full(20, 0.2),
        "j-1": np.full(20, 0.4),
        "j-2": np.full(20, 0.6),
        "j-3": np.full(20, 0.8),
    }
    result = decompose_variance(samples, alpha=0.1, n_bootstrap=300, rng=rng)
    assert result.aleatoric == pytest.approx(0.0)
    # Var of [0.2, 0.4, 0.6, 0.8] with ddof=1 = 0.0667
    assert result.epistemic == pytest.approx(0.06666, abs=1e-3)


def test_total_is_sum_of_components() -> None:
    rng = np.random.default_rng(seed=2)
    samples = {
        "j-0": rng.normal(0.3, 0.05, size=30),
        "j-1": rng.normal(0.7, 0.08, size=30),
        "j-2": rng.normal(0.5, 0.06, size=30),
    }
    result = decompose_variance(samples, alpha=0.1, n_bootstrap=200, rng=rng)
    assert result.total == pytest.approx(result.aleatoric + result.epistemic, abs=1e-9)


def test_ci_bounds_well_formed() -> None:
    rng = np.random.default_rng(seed=3)
    samples = {
        "j-0": rng.normal(0.5, 0.1, size=20),
        "j-1": rng.normal(0.5, 0.1, size=20),
    }
    result = decompose_variance(samples, alpha=0.2, n_bootstrap=300, rng=rng)
    assert result.aleatoric_ci_low <= result.aleatoric_ci_high
    assert result.epistemic_ci_low <= result.epistemic_ci_high
    assert result.aleatoric_ci_low >= 0.0
    assert result.epistemic_ci_low >= 0.0


def test_rejects_single_judge() -> None:
    with pytest.raises(ValueError, match="≥ 2 judges"):
        decompose_variance({"only": np.array([0.5, 0.6])})


def test_rejects_singleton_samples() -> None:
    with pytest.raises(ValueError, match="≥ 2 samples"):
        decompose_variance({"j1": np.array([0.5]), "j2": np.array([0.5, 0.6])})
