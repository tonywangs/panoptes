# Methods

Per-method math and citations for PANOPTES. This document tracks what is
*implemented* (status flag) and what is *planned* for later milestones. Each
section gives the underlying assumption, the estimator, and the implementation
location.

---

## Split conformal prediction — **M1, shipped**

**File**: `src/panoptes/uq/conformal_split.py`

### Assumption

The calibration set `{(X_i, Y_i)}_{i=1..n}` and the test point `(X_{n+1}, Y_{n+1})`
are *exchangeable* — strictly weaker than i.i.d. and the standard hypothesis
of distribution-free conformal prediction.

### Estimator

Let `μ̂` be a point predictor learned on a disjoint training set. Define the
calibration residuals

```
R_i = |Y_i - μ̂(X_i)|, i = 1..n
```

and let `q` be the `ceil((n+1)(1-α)) / n`-th empirical quantile of `(R_1, ..., R_n)`.
The prediction set is

```
C(x) = [μ̂(x) - q, μ̂(x) + q]
```

### Guarantee

Marginal coverage:

```
P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α
```

with no parametric assumptions beyond exchangeability. The `+1` in the
quantile rank corrects for the finite-sample worst case. When
`ceil((n+1)(1-α)) > n` (i.e. `α < 1/(n+1)`), no finite quantile achieves the
target coverage and we return `+∞` rather than silently undercovering.

### References

- Papadopoulos, Proedrou, Vovk, Gammerman (2002). *Inductive Confidence Machines for Regression.* ECML.
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World.* Springer.
- Angelopoulos, Bates (2023). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* Tutorial / arXiv:2107.07511.

---

## Adaptive / locally-weighted conformal (CQR) — **planned M2**

**File**: `src/panoptes/uq/conformal_adaptive.py` (stub)

Conformalized Quantile Regression: fit two quantile regressors `q̂_{α/2}` and
`q̂_{1-α/2}` on the training split; compute conformity scores from their
out-of-sample residuals; calibrate as in split conformal. Interval widths are
*input-adaptive*: harder examples get wider intervals.

**Reference**: Romano, Patterson, Candès (2019). *Conformalized Quantile Regression.* NeurIPS.

---

## Mondrian / group-conditional conformal — **planned M2**

**File**: `src/panoptes/uq/conformal_mondrian.py` (stub)

Partition the calibration set by `task_family` (code / math / factuality /
freeform). Compute per-group quantiles; serve each test example from its
group's quantile. Recovers *conditional* coverage under the same exchangeability
assumption, at the cost of smaller per-group calibration sets.

**Reference**: Vovk, Lindsay, Nouretdinov, Gammerman (2003). *Mondrian Confidence Machine.*

---

## Semantic entropy — **planned M2**

**File**: `src/panoptes/uq/semantic_entropy.py` (stub)

Draw `N` temperature-1 samples from the judge; cluster by *bidirectional NLI
entailment* (samples mutually entail → same cluster); compute Shannon entropy
over the cluster-size distribution. Bidirectional NLI backend: local
DeBERTa-v3-large-mnli by default, LLM-as-NLI fallback.

**Reference**: Farquhar, Kossen, Kuhn, Gal (2024). *Detecting hallucinations in large language models using semantic entropy.* Nature.

---

## Self-consistency variance — **planned M2**

**File**: `src/panoptes/uq/self_consistency.py` (stub)

Monte Carlo variance + IQR over `n` temperature samples; Bayesian bootstrap CI
on the mean using Dirichlet(1, ..., 1) weights.

**References**:
- Wang, Wei, Schuurmans, Le, Chi, et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.
- Rubin (1981). *The Bayesian Bootstrap.* Annals of Statistics.

---

## Inter-judge disagreement under a latent-ability model — **planned M3**

**File**: `src/panoptes/uq/disagreement.py` (stub)

Two aggregators depending on score scale:

- **Continuous (`scale=continuous`)**: hierarchical Gaussian model
  `score_ij = θ_i + bias_j + ε_ij,  ε_ij ~ N(0, σ_j²)`,
  fit by closed-form EM. Returns posterior mean and variance of `θ_i`, plus
  per-judge bias and precision estimates.
- **Ordinal Likert (`scale=likert_1_5`)**: ordinal Dawid-Skene with a
  smoothness prior on the confusion matrices.

**References**:
- Dawid, Skene (1979). *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* JRSS-C.
- Hovy, Berg-Kirkpatrick, Vaswani, Hovy (2013). *Learning Whom to Trust with MACE.* NAACL.

---

## Aleatoric / epistemic decomposition — **planned M3**

**File**: `src/panoptes/uq/decomposition.py` (stub)

Nested resampling:

```
Var_total ≈ E_j[Var(score | judge=j)] + Var_j[E(score | judge=j)]
            └──── aleatoric ────┘    └──── epistemic ────┘
```

Outer bootstrap over judges; inner bootstrap over temperature samples within
judge. Returned with per-example bootstrap CIs.

**References**:
- Kendall, Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS.
- Depeweg, Hernández-Lobato, Doshi-Velez, Udluft (2018). *Decomposition of Uncertainty in Bayesian Deep Learning.* ICML.

---

## Coverage diagnostics — **planned M4**

**File**: `src/panoptes/stats/coverage_tests.py` (stub)

- Marginal coverage with Clopper-Pearson CI.
- Conditional coverage per task family with Bonferroni-corrected p-values.
- Hosmer-Lemeshow binning goodness-of-fit test.

**Reference**: Hosmer, Lemeshow (1980). *Goodness-of-fit tests for the multiple logistic regression model.*

---

## Calibration metrics — **planned M4**

**File**: `src/panoptes/stats/reliability.py` (stub)

- ECE, MCE (Naeini, Cooper, Hauskrecht 2015).
- Brier score.
- Sharpness-vs-calibration framing (Gneiting & Raftery 2007).
- Reliability diagram with 95% bootstrap bands (Bröcker & Smith 2007).

---

## Bandit routing — **planned M3**

**File**: `src/panoptes/routing/bandit.py` (stub)

Thompson sampling over `Beta(α_jt, β_jt)` per `(judge, task_family)`; the
"reward" for arm `j` on item `i` is `epistemic_variance_reduction(i, j) / cost_usd(j)`
— information-per-dollar. Warm-start from M2 calibration data.

**Reference**: Russo, Van Roy, Kazerouni, Osband, Wen (2018). *A Tutorial on Thompson Sampling.* arXiv:1707.02038.
