"""Mondrian / group-conditional conformal prediction.

Split conformal gives *marginal* coverage `P(Y ∈ C(X)) >= 1 - α` over the full
calibration distribution. When the population is heterogeneous (e.g.
PANOPTES's `task_family` split into code / math / factuality / freeform),
marginal coverage can hide large per-group miscoverage. Mondrian conformal
restores *conditional* coverage by partitioning the calibration set on a
group function `g(X)` and computing a per-group conformal quantile:

    C(x) = [μ̂(x) - q_{g(x)}, μ̂(x) + q_{g(x)}],
    q_g = ceil((n_g + 1)(1 - α)) / n_g -th empirical quantile of group g's
           calibration residuals.

The marginal guarantee `P(Y ∈ C(X) | g(X) = g) >= 1 - α` holds within each
group under exchangeability *within* the group — a strictly weaker assumption
than full exchangeability across groups.

Small groups: when `n_g` is below `min_group_size`, the group's residuals are
pooled into a marginal fallback quantile. This trades a small loss in
conditional coverage for a meaningful threshold (otherwise alpha < 1/(n_g+1)
returns `+inf` and the bound is uninformative). The default 50 follows
practitioner conventions for the "rule of thumb" minimum.

References
----------
- Vovk, Lindsay, Nouretdinov, Gammerman (2003). *Mondrian Confidence Machine.*
- Romano, Sesia, Candès (2020). *Classification with Valid and Adaptive Coverage.* NeurIPS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from panoptes.uq.conformal_split import SplitConformal, split_conformal_quantile


@dataclass(slots=True)
class MondrianConformal:
    """Fitted group-conditional split conformal.

    Attributes
    ----------
    per_group : Mapping[str, SplitConformal]
        One fitted predictor per group; missing groups (unseen at fit time
        or below `min_group_size`) fall back to `marginal`.
    marginal : SplitConformal
        Pooled fallback fitted on the union of all residuals.
    min_group_size : int
        Calibration set size below which a group is considered too small
        to warrant its own quantile and is served by `marginal`.
    """

    per_group: dict[str, SplitConformal]
    marginal: SplitConformal
    min_group_size: int = 50
    score_lo: float = 0.0
    score_hi: float = 1.0

    @classmethod
    def fit(
        cls,
        predictions_by_group: Mapping[str, NDArray[np.floating]],
        labels_by_group: Mapping[str, NDArray[np.floating]],
        *,
        min_group_size: int = 50,
        score_lo: float = 0.0,
        score_hi: float = 1.0,
    ) -> MondrianConformal:
        """Build per-group and marginal residuals from disjoint group calibration sets."""
        if predictions_by_group.keys() != labels_by_group.keys():
            raise ValueError(
                "predictions_by_group and labels_by_group must have identical groups"
            )
        per_group: dict[str, SplitConformal] = {}
        all_residuals: list[NDArray[np.floating]] = []
        for group, preds in predictions_by_group.items():
            preds_arr = np.asarray(preds, dtype=np.float64)
            labs_arr = np.asarray(labels_by_group[group], dtype=np.float64)
            if preds_arr.shape != labs_arr.shape:
                raise ValueError(
                    f"group {group!r}: predictions {preds_arr.shape} and "
                    f"labels {labs_arr.shape} must match"
                )
            if preds_arr.ndim != 1:
                raise ValueError(f"group {group!r}: inputs must be 1-D")
            residuals = np.abs(preds_arr - labs_arr)
            all_residuals.append(residuals)
            if residuals.shape[0] >= min_group_size:
                per_group[group] = SplitConformal(
                    residuals=residuals, score_lo=score_lo, score_hi=score_hi
                )
        marginal_residuals = np.concatenate(all_residuals) if all_residuals else np.array([0.0])
        marginal = SplitConformal(
            residuals=marginal_residuals, score_lo=score_lo, score_hi=score_hi
        )
        return cls(
            per_group=per_group,
            marginal=marginal,
            min_group_size=min_group_size,
            score_lo=score_lo,
            score_hi=score_hi,
        )

    def predict_interval(
        self,
        point: float,
        group: str,
        *,
        alpha: float,
    ) -> tuple[float, float]:
        """Return `(lo, hi)` for `point` belonging to `group`, falling back to marginal."""
        cp = self.per_group.get(group, self.marginal)
        return cp.predict_interval(point, alpha=alpha)

    def quantile(self, group: str, *, alpha: float) -> float:
        """Conformal quantile for `group`, fallback to marginal if unseen / small."""
        cp = self.per_group.get(group)
        if cp is None:
            return split_conformal_quantile(self.marginal.residuals, alpha)
        return cp.quantile(alpha)

    def groups(self) -> list[str]:
        """Groups that have their own fitted quantile (above `min_group_size`)."""
        return sorted(self.per_group.keys())
