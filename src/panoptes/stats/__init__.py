"""Statistical diagnostics: bootstrap, coverage tests, reliability, comparison, Pareto."""

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
from panoptes.stats.coverage_tests import (
    clopper_pearson_ci,
    conditional_coverage_test,
    hosmer_lemeshow_test,
    marginal_coverage,
)
from panoptes.stats.pareto import coverage_width_pareto
from panoptes.stats.reliability import (
    brier_score,
    ece,
    mce,
    reliability_curve,
)

__all__ = [
    "bayesian_bootstrap_mean",
    "brier_score",
    "clopper_pearson_ci",
    "conditional_coverage_test",
    "coverage_width_pareto",
    "ece",
    "hosmer_lemeshow_test",
    "marginal_coverage",
    "mce",
    "paired_bootstrap_diff",
    "paired_bootstrap_kendall",
    "paired_bootstrap_spearman",
    "permutation_test_disagreement",
    "pivot_ci",
    "reliability_curve",
]
