"""Unit tests for `panoptes.uq.disagreement.HierarchicalGaussianAggregator`."""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.uq.disagreement import (
    HierarchicalGaussianAggregator,
    matrix_from_responses,
)


def test_recovers_known_means_under_zero_noise() -> None:
    """With zero judge noise and matching biases, theta = mean across judges."""
    rng = np.random.default_rng(seed=0)
    n_items = 30
    theta_true = rng.uniform(0.1, 0.9, size=n_items)
    bias_true = np.array([0.0, 0.0, 0.0])
    y = theta_true[:, None] + bias_true[None, :]  # noiseless
    agg = HierarchicalGaussianAggregator(max_iter=50, tol=1e-9)
    fit = agg.fit(y, [f"item-{i}" for i in range(n_items)], ["A", "B", "C"])
    for i in range(n_items):
        assert fit.items[i].posterior_mean == pytest.approx(theta_true[i], abs=1e-5)


def test_recovers_biases_and_sigma_ordering() -> None:
    """Biases are identifiable up to a global shift; sigmas track but are
    not perfectly recoverable from one observation per (item, judge) when
    inter-judge sigmas are very different. We verify:
      1. Biases recovered within 0.05 absolute.
      2. The judge with the largest true sigma is fitted as having the
         largest (or tied-largest) sigma.
    """
    rng = np.random.default_rng(seed=1)
    n_items = 500
    theta_true = rng.uniform(0.1, 0.9, size=n_items)
    bias_true = np.array([+0.1, 0.0, -0.1])
    sigma_true = np.array([0.05, 0.15, 0.02])
    y = (
        theta_true[:, None]
        + bias_true[None, :]
        + sigma_true[None, :] * rng.standard_normal((n_items, 3))
    )
    agg = HierarchicalGaussianAggregator()
    fit = agg.fit(y, [f"i-{i}" for i in range(n_items)], ["A", "B", "C"])
    fitted_biases = np.array([j.bias for j in fit.judges])
    fitted_sigmas = np.array([j.sigma for j in fit.judges])
    np.testing.assert_allclose(fitted_biases, bias_true, atol=0.05)
    # Judge B (true sigma 0.15) should be fitted as the noisiest.
    assert int(np.argmax(fitted_sigmas)) == 1
    # And no sigma estimate should be obviously off (e.g., > 2x truth).
    assert np.all(fitted_sigmas < 0.4)


def test_handles_missing_observations() -> None:
    """NaN entries mean a judge did not score that item; fit must ignore them."""
    rng = np.random.default_rng(seed=2)
    n_items = 100
    theta_true = rng.uniform(0.1, 0.9, size=n_items)
    y = np.column_stack(
        [
            theta_true + 0.05 * rng.standard_normal(n_items),
            theta_true + 0.05 * rng.standard_normal(n_items),
        ]
    )
    # Drop half of judge B's observations.
    y[::2, 1] = np.nan
    agg = HierarchicalGaussianAggregator()
    fit = agg.fit(y, [f"i-{i}" for i in range(n_items)], ["A", "B"])
    assert fit.converged or fit.n_iter > 0
    posterior = np.array([it.posterior_mean for it in fit.items])
    # Should still track the true theta with reasonable accuracy.
    rmse = float(np.sqrt(((posterior - theta_true) ** 2).mean()))
    assert rmse < 0.1


def test_rejects_single_judge() -> None:
    with pytest.raises(ValueError, match="≥ 2 judges"):
        HierarchicalGaussianAggregator().fit(
            np.array([[0.5], [0.6]]), ["a", "b"], ["only"]
        )


def test_matrix_from_responses() -> None:
    nested = {
        "item-1": {"judge-A": 0.7, "judge-B": 0.6},
        "item-2": {"judge-A": 0.3},
    }
    matrix, items, judges = matrix_from_responses(nested)
    assert items == ["item-1", "item-2"]
    assert judges == ["judge-A", "judge-B"]
    assert matrix[0, 0] == 0.7
    assert matrix[0, 1] == 0.6
    assert matrix[1, 0] == 0.3
    assert np.isnan(matrix[1, 1])
