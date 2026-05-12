"""Uncertainty-quantification methods for PANOPTES.

Conformal prediction (split, adaptive/CQR, Mondrian group-conditional),
semantic entropy over temperature samples, self-consistency variance,
hierarchical-Gaussian jury aggregation, aleatoric/epistemic decomposition.
See METHODS.md for the math and citations.
"""

from panoptes.uq.conformal_split import SplitConformal, split_conformal_quantile

__all__ = ["SplitConformal", "split_conformal_quantile"]
