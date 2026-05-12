# Methods

Per-method math and citations for PANOPTES. Each section gives the underlying assumption, the estimator, and the implementation location.

---

## Split conformal prediction

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

## Adaptive / locally-weighted conformal (CQR)

**File**: `src/panoptes/uq/conformal_adaptive.py`

Conformalized Quantile Regression: fit two quantile regressors `q̂_{α/2}` and
`q̂_{1-α/2}` on the training split (we use `GradientBoostingRegressor(loss='quantile')`);
compute the signed conformity scores

```
E_i = max(q̂_{α/2}(X_i) - Y_i, Y_i - q̂_{1-α/2}(X_i))
```

on the calibration split; the interval is `[q̂_{α/2}(x) - Q_E, q̂_{1-α/2}(x) + Q_E]`
where `Q_E = ceil((n+1)(1-α))/n`-th quantile of `{E_i}`. Interval widths are
*input-adaptive*: low-noise regions get narrower intervals than high-noise
ones, while the marginal coverage `1 - α` guarantee is preserved.

**Reference**: Romano, Patterson, Candès (2019). *Conformalized Quantile Regression.* NeurIPS.

---

## Mondrian / group-conditional conformal

**File**: `src/panoptes/uq/conformal_mondrian.py`

Partition the calibration set by `task_family` (code / math / factuality /
freeform). Compute per-group quantiles; serve each test example from its
group's quantile. Recovers *conditional* coverage:

```
P(Y ∈ C(X) | g(X) = g) ≥ 1 - α  for every group g with n_g ≥ min_group_size
```

at the cost of smaller per-group calibration sets. Groups below
`min_group_size` (default 50) fall back to the pooled marginal quantile.

**Reference**: Vovk, Lindsay, Nouretdinov, Gammerman (2003). *Mondrian Confidence Machine.*

---

## Semantic entropy

**File**: `src/panoptes/uq/semantic_entropy.py`

Draw `N` temperature-1 samples from the judge; cluster by *bidirectional NLI
entailment* (samples mutually entail → same cluster); compute Shannon
entropy over the cluster-size distribution `p_c = |c| / N`:

```
H = -Σ_c p_c log p_c,   bounded in [0, log N].
```

Optional log-probability weighting (`weights[c] ∝ Σ_{s ∈ c} exp(log_p(s))`)
matches the Farquhar et al. formulation when judge response log-probs are
exposed by the provider. Bidirectional NLI backends:

- **DeBERTa-v3-large-mnli** (local HF, default) — `nli/deberta.py`, behind
  the `providers-hf` extra to keep the base install lean.
- **LLM-as-NLI** fallback — `nli/llm.py`, uses any `LLMClient` and asks for
  a 3-way label via structured output. Costs O(N²) LLM calls per item.

**Reference**: Farquhar, Kossen, Kuhn, Gal (2024). *Detecting hallucinations in large language models using semantic entropy.* Nature.

---

## Self-consistency variance

**File**: `src/panoptes/uq/self_consistency.py`

For `n` temperature-sampled scores from the same `(judge, item)` pair:

- **Sample variance** with `ddof=1` (unbiased)
- **IQR** via `scipy.stats.iqr`
- **Bayesian bootstrap CI** on the mean (Rubin 1981): draw `B` weight
  vectors `w^(b) ~ Dirichlet(1, ..., 1)`, form weighted means
  `μ^(b) = Σ w^(b)_i s_i`, and report the `(α/2, 1-α/2)` quantiles of
  `{μ^(b)}`.

The Bayesian bootstrap (vs Efron) avoids ties from discrete resampling and
matches the posterior of a noninformative Dirichlet-process prior, which is
more honest at the small `n` we typically see (≤ 20 samples per item).

**References**:
- Wang, Wei, Schuurmans, Le, Chi, et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.
- Rubin (1981). *The Bayesian Bootstrap.* Annals of Statistics.

---

## Inter-judge disagreement under a latent-ability model

**File**: `src/panoptes/uq/disagreement.py`

The continuous case is the primary path for PANOPTES (rubric scores live in
`[0, 1]`). The model is

```
score_ij = θ_i + b_j + ε_ij,   ε_ij ~ N(0, σ_j²),   Σ_j b_j = 0
```

fit by closed-form EM. The M-step for `σ_j²` includes the standard
hierarchical-Gaussian correction

```
σ_j² = E[(y_ij - θ_i - b_j)² | data]
     = (y_ij - μ_i_post - b_j)² + Var[θ_i | data]
```

without which low-noise judges have their `σ_j` underestimated. The
`Σ_j b_j = 0` constraint resolves the obvious additive identifiability
between θ and bias.

Outputs:
- per-item `posterior_mean`, `posterior_var` (the latent quality and its
  uncertainty);
- per-judge `bias`, `sigma`, `precision`.

For ordinal Likert (1–5) rubrics, scores are mapped onto `[0, 1]` via
`(value - 1) / 4` and aggregated with the same continuous hierarchical
Gaussian.

**References**:
- Dawid, Skene (1979). *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* JRSS-C.
- Hovy, Berg-Kirkpatrick, Vaswani, Hovy (2013). *Learning Whom to Trust with MACE.* NAACL.
- Bishop (2006). *Pattern Recognition and Machine Learning*, §10.

---

## Aleatoric / epistemic decomposition

**File**: `src/panoptes/uq/decomposition.py`

Law of total variance applied to the `(judge, item)` system:

```
Var_total = E_j[Var(score | judge=j)]  +  Var_j[E(score | judge=j)]
            └──── aleatoric ────┘         └──── epistemic ────┘
```

Estimator: per-judge sample mean `m_j` and within-judge variance `v_j`;
`aleatoric = sample-weighted mean of v_j`, `epistemic = sample variance of
{m_j}`. Bootstrap CIs by nested resampling (outer over judges, inner over
temperature samples within judge) at the requested `α`.

**References**:
- Kendall, Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS.
- Depeweg, Hernández-Lobato, Doshi-Velez, Udluft (2018). *Decomposition of Uncertainty in Bayesian Deep Learning.* ICML.

---

## Coverage diagnostics

**File**: `src/panoptes/stats/coverage_tests.py`

- **Marginal coverage** with **Clopper-Pearson** exact binomial CI
  (Clopper & Pearson 1934). Returned at the requested `1 - α`.
- **Conditional coverage** per group (typically `task_family`): per-group
  Clopper-Pearson + two-sided exact binomial p-values for
  H0: `P(cover|group) = target`, **Bonferroni-corrected** across the K
  groups so the family-wise error rate stays ≤ α.
- **Hosmer-Lemeshow** binning test (Hosmer & Lemeshow 1980): predictions
  binned by score, χ² statistic accumulated from `(O - E)² / E(1-E)·n_b`
  with df = `n_bins - 2`.

**References**:
- Clopper, Pearson (1934). *The use of confidence or fiducial limits illustrated in the case of the binomial.* Biometrika.
- Hosmer, Lemeshow (1980). *Goodness-of-fit tests for the multiple logistic regression model.*
- Vovk (2012). *Conditional Validity of Inductive Conformal Predictors.*

---

## Calibration metrics

**File**: `src/panoptes/stats/reliability.py`

- **ECE** (Expected Calibration Error): bin-weighted L1 gap between
  in-bin confidence and accuracy. Default 15 bins per Guo et al. 2017.
- **MCE** (Maximum Calibration Error): worst-case bin gap.
- **Brier score**: mean squared error between probabilistic prediction
  and `{0, 1}` label.
- **Reliability curve** with optional 95% bootstrap bands: per-bin
  observed accuracy across paired bootstrap resamples of
  `(predictions, labels)`. Wider than pointwise binomial CIs because the
  bands include bin-mean prediction variability, per Bröcker & Smith (2007).

Sharpness vs calibration is the Gneiting-Raftery framing we use to
present trade-offs in the dashboard: sharp predictions cluster near
0/1, calibrated predictions match empirical frequencies. The two are
optimized jointly via proper scoring rules.

**References**:
- Naeini, Cooper, Hauskrecht (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI.
- Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML.
- Gneiting, Raftery (2007). *Strictly Proper Scoring Rules.* JASA.
- Bröcker, Smith (2007). *Increasing the Reliability of Reliability Diagrams.* Weather and Forecasting.

---

## Routing strategies

**File**: `src/panoptes/routing/*`

Strategies are all routed through the `JuryRouter` Protocol, which the
pipeline calls per item to decide which judges to invoke:

- `AllJudges` — baseline; calls every available judge.
- `SingleJudge` — calls one judge (cheapest tier by default, or a named one).
- `EscalationPolicy` — calls all cheap-tier judges first; if the
  inter-judge variance exceeds `tau`, escalates to one expensive judge.
- `ThompsonBandit` — Beta(α, β) per `(judge, task_family)`; the reward
  signal for arm `j` on item `i` is

  ```
  reward_j = 1[ info_per_dollar(j, i) ≥ median over called judges ]
  info_per_dollar(j, i) = max(0, var_before - var_after_excluding_j) / cost_usd(j)
  ```

  i.e. did this judge meaningfully tighten the variance per dollar relative
  to the median called judge on this item? The state is JSON-serializable so
  the bandit can warm-start across runs.

**References**:
- Russo, Van Roy, Kazerouni, Osband, Wen (2018). *A Tutorial on Thompson Sampling.* arXiv:1707.02038.
- Chapelle, Li (2011). *An Empirical Evaluation of Thompson Sampling.* NeurIPS.
