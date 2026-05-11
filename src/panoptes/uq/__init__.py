"""Uncertainty-quantification methods for PANOPTES.

M1 ships split conformal prediction. M2 adds adaptive (CQR) and Mondrian
variants, semantic entropy over temperature samples, and self-consistency
variance. M3 adds inter-judge disagreement aggregation and the aleatoric/
epistemic decomposition. See METHODS.md for the math and citations.
"""

from panoptes.uq.conformal_split import SplitConformal, split_conformal_quantile

__all__ = ["SplitConformal", "split_conformal_quantile"]
