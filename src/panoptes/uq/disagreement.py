"""Hierarchical-Gaussian jury aggregation under a latent-ability model.

The headline aggregator for PANOPTES's continuous-score path. Naive averaging
treats every judge as equally trustworthy and centered; in practice judges
have *bias* (one judge runs hot, another cold) and differing *precision*
(one judge is consistent, another is noisy). A hierarchical Gaussian model
captures both:

    score_ij = θ_i + b_j + ε_ij,   ε_ij ~ N(0, σ_j²)

where:
    - `θ_i` is the latent true score of item i (the quantity we want);
    - `b_j` is judge j's systematic bias (Σ_j b_j = 0 by convention, the
      sign-flip symmetry that gives identifiability);
    - `σ_j²` is judge j's noise variance.

Inference is by closed-form EM (Dempster, Laird, Rubin 1977 in general; for
this specific Gaussian random-effects model the updates are tractable in
closed form and converge in a handful of iterations):

E-step (posterior of θ_i given current b_j, σ_j):
    τ_j     = 1 / σ_j²
    P_i     = Σ_j τ_j                    (posterior precision)
    μ_post_i= (Σ_j τ_j (y_ij - b_j)) / P_i

M-step:
    b_j     = mean_i (y_ij - μ_post_i)    (re-centered so Σ b_j = 0)
    σ_j²    = mean_i ((y_ij - μ_post_i - b_j)²)

This is essentially a one-way Gaussian random-effects model; equivalent to
(a) a Bayesian model with non-informative priors on b_j, σ_j, or (b) a
restricted-likelihood (REML) fit. We use plain ML / EM since priors are
not required for PANOPTES's use case and REML adds complexity without
material benefit at the small `(I, J)` we operate on.

For ordinal-categorical labels (e.g. 1–5 Likert), PANOPTES will ship a
Dawid-Skene ordinal aggregator in M5; the M3 release covers the continuous
case which is what `RubricScore.value` is.

References
----------
- Dawid, Skene (1979). *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* JRSS-C.
- Hovy, Berg-Kirkpatrick, Vaswani, Hovy (2013). *Learning Whom to Trust With MACE.* NAACL.
- Bishop (2006). *Pattern Recognition and Machine Learning*, §10 (EM for mixture models, applicable here as a degenerate one-cluster mixture).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class JudgeParams:
    """Estimated per-judge parameters after fitting."""

    judge_id: str
    bias: float
    sigma: float

    @property
    def precision(self) -> float:
        return 1.0 / (self.sigma**2 + 1e-12)


@dataclass(frozen=True, slots=True)
class ItemPosterior:
    """Posterior over `θ_i` for one item."""

    item_id: str
    posterior_mean: float
    posterior_var: float


@dataclass(slots=True)
class HierarchicalGaussianFit:
    """Result of fitting `HierarchicalGaussianAggregator`."""

    judges: list[JudgeParams]
    items: list[ItemPosterior]
    n_iter: int
    converged: bool

    def judge_by_id(self) -> dict[str, JudgeParams]:
        return {j.judge_id: j for j in self.judges}

    def item_by_id(self) -> dict[str, ItemPosterior]:
        return {it.item_id: it for it in self.items}


class HierarchicalGaussianAggregator:
    """Closed-form EM fit of the hierarchical Gaussian model.

    Usage
    -----
    Build a 2-D score matrix `Y[i, j]` (I items × J judges) with NaN where a
    judge did not score an item, then `fit(Y, item_ids, judge_ids)`.
    """

    def __init__(
        self,
        *,
        max_iter: int = 50,
        tol: float = 1e-6,
        min_sigma: float = 1e-3,
    ) -> None:
        self._max_iter = max_iter
        self._tol = tol
        self._min_sigma = min_sigma

    def fit(
        self,
        y: NDArray[np.floating],
        item_ids: list[str],
        judge_ids: list[str],
    ) -> HierarchicalGaussianFit:
        """Fit the hierarchical Gaussian via EM.

        Parameters
        ----------
        y : (I, J) array, possibly NaN for missing observations
            Score matrix in [0, 1].
        item_ids : list of length I
        judge_ids : list of length J
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"y must be 2-D (items × judges); got shape {y.shape}")
        n_items, n_judges = y.shape
        if len(item_ids) != n_items:
            raise ValueError(
                f"item_ids has length {len(item_ids)} but y has {n_items} rows"
            )
        if len(judge_ids) != n_judges:
            raise ValueError(
                f"judge_ids has length {len(judge_ids)} but y has {n_judges} cols"
            )
        if n_judges < 2:
            raise ValueError(
                f"hierarchical aggregation needs ≥ 2 judges; got {n_judges}"
            )

        mask = ~np.isnan(y)  # observed mask
        # Initialization
        with np.errstate(invalid="ignore"):
            theta = np.where(mask, y, 0.0).sum(axis=1) / mask.sum(axis=1).clip(min=1)
        bias = np.zeros(n_judges)
        sigma = np.full(n_judges, 0.1)
        posterior_var = np.full(n_items, np.nan, dtype=np.float64)
        converged = False
        n_iter = 0
        for it in range(self._max_iter):
            n_iter = it + 1
            theta_prev = theta.copy()
            # E-step: posterior over θ_i given current (bias, sigma).
            tau = 1.0 / np.maximum(sigma**2, self._min_sigma**2)
            # Per-item: precision = Σ_j τ_j * mask_ij
            precision = (mask.astype(np.float64) * tau).sum(axis=1)
            # Per-item: numerator = Σ_j τ_j * (y_ij - bias_j) * mask_ij
            centered = np.where(mask, y - bias, 0.0)
            numer = (centered * tau).sum(axis=1)
            theta = numer / np.maximum(precision, 1e-12)
            posterior_var = 1.0 / np.maximum(precision, 1e-12)

            # M-step: update bias_j and sigma_j².
            # σ_j² = E[(y_ij - θ_i - b_j)² | data]
            #     = (y_ij - μ_i_post - b_j)² + Var[θ_i | data]
            # The second term accounts for the uncertainty in θ — omitting it
            # is the classic EM-for-Gaussian-random-effects bias that drives
            # σ_j estimates downward for low-noise judges.
            new_bias = np.zeros(n_judges)
            new_sigma = np.zeros(n_judges)
            for j in range(n_judges):
                col = mask[:, j]
                if not col.any():
                    new_bias[j] = bias[j]
                    new_sigma[j] = sigma[j]
                    continue
                resid_no_bias = y[col, j] - theta[col]
                new_bias[j] = float(resid_no_bias.mean())
                resid = resid_no_bias - new_bias[j]
                # Add posterior variance per observed item.
                second_moment = float(
                    ((resid**2) + posterior_var[col]).mean()
                )
                new_sigma[j] = max(self._min_sigma, float(np.sqrt(second_moment)))
            # Center biases so Σ_j b_j = 0 — required for identifiability,
            # since (θ + c, b - c) is observationally equivalent to (θ, b).
            mean_bias = float(new_bias.mean())
            new_bias = new_bias - mean_bias
            theta = theta + mean_bias

            delta = float(np.max(np.abs(theta - theta_prev)))
            bias = new_bias
            sigma = new_sigma
            if delta < self._tol:
                converged = True
                break

        return HierarchicalGaussianFit(
            judges=[
                JudgeParams(judge_id=jid, bias=float(bias[j]), sigma=float(sigma[j]))
                for j, jid in enumerate(judge_ids)
            ],
            items=[
                ItemPosterior(
                    item_id=iid,
                    posterior_mean=float(theta[i]),
                    posterior_var=float(posterior_var[i]),
                )
                for i, iid in enumerate(item_ids)
            ],
            n_iter=n_iter,
            converged=converged,
        )


def matrix_from_responses(
    judge_responses_by_item: dict[str, dict[str, float]],
) -> tuple[NDArray[np.floating], list[str], list[str]]:
    """Build a `(I × J)` score matrix from a nested-dict view of responses.

    Items / judges not present become `NaN`. The returned `item_ids` /
    `judge_ids` lists give the row / column labels in stable order so the
    caller can join the fitted parameters back to source records.
    """
    item_ids = sorted(judge_responses_by_item.keys())
    judge_set: set[str] = set()
    for inner in judge_responses_by_item.values():
        judge_set.update(inner.keys())
    judge_ids = sorted(judge_set)
    matrix = np.full((len(item_ids), len(judge_ids)), np.nan, dtype=np.float64)
    j_index = {jid: j for j, jid in enumerate(judge_ids)}
    for i, iid in enumerate(item_ids):
        for jid, value in judge_responses_by_item[iid].items():
            matrix[i, j_index[jid]] = value
    return matrix, item_ids, judge_ids
