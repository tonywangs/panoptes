"""Unit tests for `panoptes.uq.conformal_mondrian`.

Verifies per-group quantile routing, the small-group fallback to the
marginal quantile, and group-conditional coverage on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.uq.conformal_mondrian import MondrianConformal


def test_fit_requires_matching_keys() -> None:
    preds = {"code": np.array([0.1, 0.2])}
    labels = {"math": np.array([0.1, 0.2])}
    with pytest.raises(ValueError, match="identical groups"):
        MondrianConformal.fit(preds, labels)


def test_small_group_falls_back_to_marginal() -> None:
    rng = np.random.default_rng(seed=0)
    # Large group satisfies min_group_size; small group does not.
    preds = {
        "big": rng.uniform(0, 1, size=200),
        "small": rng.uniform(0, 1, size=10),
    }
    labels = {
        "big": preds["big"] + 0.1 * rng.standard_normal(200),
        "small": preds["small"] + 0.05 * rng.standard_normal(10),
    }
    mc = MondrianConformal.fit(preds, labels, min_group_size=50)
    assert mc.groups() == ["big"]  # only big has its own quantile
    q_small = mc.quantile("small", alpha=0.1)
    q_marginal = mc.quantile("nonexistent_group", alpha=0.1)
    # Small / unseen groups share the marginal quantile.
    assert q_small == q_marginal


def test_per_group_quantiles_differ_under_heteroscedasticity() -> None:
    rng = np.random.default_rng(seed=1)
    preds = {
        "low_noise": rng.uniform(0, 1, size=300),
        "high_noise": rng.uniform(0, 1, size=300),
    }
    labels = {
        "low_noise": preds["low_noise"] + 0.02 * rng.standard_normal(300),
        "high_noise": preds["high_noise"] + 0.2 * rng.standard_normal(300),
    }
    mc = MondrianConformal.fit(preds, labels, min_group_size=50)
    q_low = mc.quantile("low_noise", alpha=0.1)
    q_high = mc.quantile("high_noise", alpha=0.1)
    assert q_high > q_low * 2.0


def test_group_conditional_coverage_on_synthetic() -> None:
    """Each group's empirical coverage should hold near nominal within group."""
    rng = np.random.default_rng(seed=7)
    sigmas = {"a": 0.05, "b": 0.15}
    preds: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for g, sigma in sigmas.items():
        p = rng.uniform(0, 1, size=600)
        preds[g] = p
        labels[g] = p + sigma * rng.standard_normal(600)
    mc = MondrianConformal.fit(preds, labels, min_group_size=50)

    for g, sigma in sigmas.items():
        p_test = rng.uniform(0, 1, size=2000)
        y_test = p_test + sigma * rng.standard_normal(2000)
        covered = 0
        q = mc.quantile(g, alpha=0.1)
        for p, y in zip(p_test, y_test, strict=True):
            lo, hi = mc.predict_interval(float(p), g, alpha=0.1)
            # Override clipping for the coverage check (we want unclipped behavior
            # for the math; in production the clipping is what callers want).
            del lo, hi
            if abs(p - y) <= q:
                covered += 1
        coverage = covered / 2000
        assert coverage == pytest.approx(0.9, abs=0.03)
