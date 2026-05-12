"""Reliability metrics: ECE, MCE, Brier, reliability diagram with bootstrap bands.

Calibration speaks to whether *predicted probabilities* match *empirical
frequencies*. PANOPTES's rubric scores can be interpreted probabilistically
(e.g. "0.7 ≈ 70% chance the candidate is correct") provided you binarize
the ground truth — in M5 this means HumanEval pass/fail, in general it
means whatever boolean correctness signal the benchmark exposes.

References
----------
- Naeini, Cooper, Hauskrecht (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI.
- Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML (ECE refined).
- Gneiting, Raftery (2007). *Strictly Proper Scoring Rules.* JASA (sharpness vs calibration framing).
- Bröcker, Smith (2007). *Increasing the Reliability of Reliability Diagrams.* Weather and Forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """One reliability curve with optional bootstrap bands.

    `bin_confidence` is the mean predicted score within each bin (x-axis);
    `bin_accuracy` is the empirical fraction of correct labels (y-axis).
    `band_low`/`band_high` are bootstrap-percentile bands when computed.
    """

    bin_centers: NDArray[np.floating]
    bin_confidence: NDArray[np.floating]
    bin_accuracy: NDArray[np.floating]
    bin_counts: NDArray[np.intp]
    band_low: NDArray[np.floating] | None
    band_high: NDArray[np.floating] | None
    n_bins: int


def _bin_indices(
    predictions: NDArray[np.floating], n_bins: int
) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
    """Return per-prediction bin index and the bin centers."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # `np.clip` keeps the upper-edge value in the last bin.
    idx = np.clip(np.digitize(predictions, edges) - 1, 0, n_bins - 1)
    return idx.astype(np.intp), centers


def ece(
    predictions: NDArray[np.floating],
    labels: NDArray[np.bool_],
    *,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error: bin-weighted L1 gap between confidence and accuracy.

    `predictions` ∈ [0, 1]; `labels` ∈ {0, 1}. Defaults to 15 bins per Guo
    et al. (2017) — fewer bins are noisier, more bins introduce empty bins
    that contribute 0 to the weighted average.
    """
    arr_pred = np.asarray(predictions, dtype=np.float64)
    arr_lab = np.asarray(labels).astype(np.float64)
    if arr_pred.shape != arr_lab.shape:
        raise ValueError(
            f"predictions {arr_pred.shape} and labels {arr_lab.shape} must match"
        )
    if arr_pred.ndim != 1:
        raise ValueError(f"inputs must be 1-D; got {arr_pred.shape}")
    n = arr_pred.shape[0]
    if n == 0:
        return 0.0
    idx, _ = _bin_indices(arr_pred, n_bins)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        conf = float(arr_pred[mask].mean())
        acc = float(arr_lab[mask].mean())
        total += (n_b / n) * abs(conf - acc)
    return total


def mce(
    predictions: NDArray[np.floating],
    labels: NDArray[np.bool_],
    *,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error: worst-case bin gap. Same binning as `ece`."""
    arr_pred = np.asarray(predictions, dtype=np.float64)
    arr_lab = np.asarray(labels).astype(np.float64)
    if arr_pred.shape != arr_lab.shape:
        raise ValueError("predictions and labels must match in shape")
    if arr_pred.size == 0:
        return 0.0
    idx, _ = _bin_indices(arr_pred, n_bins)
    worst = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = float(arr_pred[mask].mean())
        acc = float(arr_lab[mask].mean())
        worst = max(worst, abs(conf - acc))
    return worst


def brier_score(
    predictions: NDArray[np.floating],
    labels: NDArray[np.bool_],
) -> float:
    """Mean squared error between probabilistic prediction and {0,1} label."""
    arr_pred = np.asarray(predictions, dtype=np.float64)
    arr_lab = np.asarray(labels).astype(np.float64)
    if arr_pred.shape != arr_lab.shape:
        raise ValueError("predictions and labels must match in shape")
    return float(((arr_pred - arr_lab) ** 2).mean())


def reliability_curve(
    predictions: NDArray[np.floating],
    labels: NDArray[np.bool_],
    *,
    n_bins: int = 15,
    n_bootstrap: int = 0,
    alpha: float = 0.1,
    rng: np.random.Generator | None = None,
) -> ReliabilityCurve:
    """Build a reliability curve. If `n_bootstrap > 0`, returns bootstrap bands.

    The bands are *per-bin* percentile bands across paired bootstrap
    resamples of `(predictions, labels)`. They are wider than pointwise
    binomial CIs (because they include bin-mean prediction variability)
    and are what Bröcker & Smith (2007) recommend for reporting.
    """
    arr_pred = np.asarray(predictions, dtype=np.float64)
    arr_lab = np.asarray(labels).astype(bool)
    if arr_pred.shape != arr_lab.shape:
        raise ValueError("predictions and labels must match")
    idx, centers = _bin_indices(arr_pred, n_bins)
    bin_confidence = np.zeros(n_bins, dtype=np.float64)
    bin_accuracy = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.intp)
    for b in range(n_bins):
        mask = idx == b
        n_b = int(mask.sum())
        bin_counts[b] = n_b
        if n_b > 0:
            bin_confidence[b] = float(arr_pred[mask].mean())
            bin_accuracy[b] = float(arr_lab[mask].mean())

    band_low: NDArray[np.floating] | None = None
    band_high: NDArray[np.floating] | None = None
    if n_bootstrap > 0:
        rand = rng if rng is not None else np.random.default_rng()
        n = arr_pred.shape[0]
        boot = np.full((n_bootstrap, n_bins), np.nan, dtype=np.float64)
        for k in range(n_bootstrap):
            pick = rand.integers(0, n, size=n)
            p_b = arr_pred[pick]
            l_b = arr_lab[pick]
            idx_b, _ = _bin_indices(p_b, n_bins)
            for b in range(n_bins):
                mask = idx_b == b
                if mask.any():
                    boot[k, b] = float(l_b[mask].mean())
        band_low = np.nanquantile(boot, alpha / 2.0, axis=0)
        band_high = np.nanquantile(boot, 1.0 - alpha / 2.0, axis=0)

    return ReliabilityCurve(
        bin_centers=centers,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_counts=bin_counts,
        band_low=band_low,
        band_high=band_high,
        n_bins=n_bins,
    )
