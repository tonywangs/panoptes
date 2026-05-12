# PANOPTES

**Uncertainty-aware LLM evaluation.** Calibrated posteriors. Aleatoric/epistemic decomposition. Conformal prediction.

PANOPTES treats LLM evaluation as a first-class statistical inference problem rather than a "score-and-stop" exercise. For every `(task, response, judge)` tuple, the framework asks: what is the posterior distribution over the true quality, and what fraction of that uncertainty is reducible (epistemic) vs irreducible (aleatoric)? That decomposition tells you whether to trust the score, sample more judges, escalate to a stronger one, or accept the response is genuinely ambiguous.

## What's in the box

**Provider-agnostic async clients**

- Anthropic, OpenAI, Google Gemini, and OpenAI-compatible (Together, Groq, vLLM) — each implemented against a single `LLMClient` Protocol.
- `httpx.AsyncClient` with per-provider `asyncio.Semaphore` rate limiting, exponential-backoff-with-full-jitter retry, and retriable-vs-terminal error taxonomy.
- Prompt-cache aware on Anthropic; provider-managed caching on OpenAI; per-call USD cost accounting normalized across providers.

**Uncertainty quantification**

- **Split conformal prediction** (Vovk/Gammerman/Shafer 2005) with finite-sample `ceil((n+1)(1-α))/n` quantile correction.
- **Conformalized Quantile Regression** (Romano, Patterson, Candès 2019) — input-adaptive widths via sklearn quantile gradient boosting.
- **Mondrian / group-conditional conformal** (Vovk et al. 2003) — per-group quantiles, fallback to marginal when group is small.
- **Semantic entropy** (Farquhar et al. *Nature* 2024) — bidirectional NLI clustering with two backends: local DeBERTa-v3-mnli and an LLM-as-NLI fallback.
- **Self-consistency** (Wang et al. 2023) — Monte Carlo variance + IQR + Bayesian bootstrap CI (Rubin 1981, Dirichlet(1,...,1) weights).
- **Hierarchical-Gaussian jury aggregation** — closed-form EM for `score_ij = θ_i + bias_j + ε_ij`, recovering per-item posteriors and per-judge bias/precision with the standard hierarchical M-step correction for σ.
- **Aleatoric/epistemic decomposition** (Kendall & Gal 2017) via nested bootstrap — outer over judges (epistemic), inner over temperature samples (aleatoric).

**Routing**

- `JuryRouter` Protocol with four strategies: `all`, `single`, `escalation` (cheap-first, escalate on high inter-judge variance), and `bandit` (Thompson sampling on Beta(α,β) per `(judge, task_family)`, reward = info-per-dollar vs median).

**Diagnostics**

- Marginal coverage with Clopper-Pearson CI, conditional-per-group with Bonferroni p-values, Hosmer-Lemeshow binning.
- ECE, MCE, Brier (Naeini et al. 2015, Guo et al. 2017), reliability diagram with 95% bootstrap bands (Bröcker & Smith 2007).
- Paired-bootstrap Spearman ρ / Kendall τ between judges; permutation test for "judges A and B disagree more than chance".
- Coverage-width Pareto sweep over α.

**Benchmarks**

- HumanEval, MBPP, GSM8K (with parsed `#### N` final-answer markers), MT-Bench, TruthfulQA (parquet from HF). All cached via `http_fetch_cached`.
- **Calibration probe** that mechanically obfuscates HumanEval — entry-point functions get renamed and tests rewritten so judges can't recall the canonical solution. Ground truth comes from sandboxed Python execution.

**Sandboxed execution**

- `subprocess`-based Python sandbox with `resource.setrlimit` CPU+memory caps and hard wall timeout. Bounds accidents and crashes; not hardened for adversarial code.

**Persistence + UI**

- DuckDB result store partitioned by `(task_family, judge_id, prompt_version_hash)` with a `judge_uq_results` sidecar for per-(item, judge, method) metric blobs.
- `panoptes report --db <duckdb> --out report.html` — self-contained offline HTML.
- Streamlit dashboard — Overview / Drill-down / Judge comparison / Conformal Pareto pages, reads DuckDB directly via `st.cache_data`.

## Install

```sh
uv sync --extra dev
```

Optional extras: `viz` (Streamlit dashboard), `providers-openai`, `providers-google`, `providers-hf` (local DeBERTa NLI), `bench` (sklearn + datasets + statsmodels + rank-bm25).

Set provider keys in a `.env` next to `pyproject.toml`. See `.env.example` for the full list.

## Quick start

Run a smoke with the deterministic mock client (no API key needed):

```sh
uv run panoptes eval humaneval --judges claude --uq split --n 5 --mock --out runs/smoke.duckdb
```

Multi-judge, sampling-based UQ, LLM-as-NLI semantic entropy:

```sh
uv run panoptes eval humaneval \
  --judges claude,gpt,gemini \
  --uq split,self-consistency,semantic-entropy \
  --n 5 --n-samples 5 --nli llm --mock \
  --out runs/m_smoke.duckdb
```

With real provider keys:

```sh
uv run panoptes eval humaneval \
  --judges claude,gpt,gemini \
  --uq split,self-consistency,semantic-entropy,decomposition \
  --strategy bandit --n 100 --n-samples 10 \
  --out runs/v1.duckdb
uv run panoptes report --db runs/v1.duckdb --out report.html
uv run streamlit run src/panoptes/dashboard/app.py -- --db runs/v1.duckdb
```

## Results

> TODO: measure. Target marginal coverage within 2pp of nominal on ≥3 of 4 conformal methods on the calibration probe, and a 30%+ bandit-vs-all-judges cost reduction on cost-per-correctly-flagged-uncertain example.

## Development

```sh
uv run ruff check .
uv run pyright
uv run pytest -q
```

CI gates on ruff, pyright `--strict`, and `--cov-fail-under=80` for `panoptes.uq` and `panoptes.stats`.

See `METHODS.md` for the statistical methods and citations, and `CONTRIBUTING.md` for design principles.

## License

MIT (see `LICENSE` once added).
