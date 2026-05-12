# PANOPTES

**Uncertainty-aware LLM evaluation.** Calibrated posteriors. Aleatoric/epistemic decomposition. Conformal prediction.

PANOPTES treats LLM evaluation as a first-class statistical inference problem rather than a "score-and-stop" exercise. For every `(task, response, judge)` tuple, the framework asks: what is the posterior distribution over the true quality, and what fraction of that uncertainty is reducible (epistemic) vs irreducible (aleatoric)? That decomposition tells you whether to trust the score, sample more judges, escalate to a stronger one, or accept the response is genuinely ambiguous.

> Status: **v1.0 milestones M1–M5 shipped.** Full roadmap below; benchmark numbers under "Results" are deliberately empty until measured on a real run.

## What's shipped (M1 + M2)

**M1 — foundation**

- Async, provider-agnostic `LLMClient` Protocol with exponential-backoff-with-full-jitter retry, retriable vs terminal error taxonomy, and per-provider `asyncio.Semaphore` rate limiting.
- Pydantic v2 schemas for every cross-module contract. `pyright --strict` clean.
- Single-call rubric judge with versioned, content-hashed prompt templates.
- **Split conformal prediction** (Vovk/Gammerman/Shafer 2005) with finite-sample `ceil((n+1)(1-α))/n` quantile correction.
- HumanEval loader, DuckDB result store partitioned by `(task_family, judge_id, prompt_version_hash)`.

**M2 — UQ breadth + remaining providers**

- Provider impls for **Anthropic**, **OpenAI**, **Google Gemini**, and **OpenAI-compatible** (Together, Groq, vLLM) — each tested with respx-mocked HTTP.
- **Conformalized Quantile Regression** (Romano, Patterson, Candès 2019) — input-adaptive interval widths via sklearn quantile gradient boosting.
- **Mondrian / group-conditional conformal** (Vovk et al. 2003) — per-group quantiles, fallback to marginal when group is small.
- **Semantic entropy** (Farquhar et al. 2024) — bidirectional NLI clustering with two backends: local **DeBERTa-v3-mnli** (`--nli=deberta`, behind the `providers-hf` extra) and **LLM-as-NLI** fallback (`--nli=llm`).
- **Self-consistency** (Wang et al. 2023) — Monte Carlo variance + IQR + Bayesian bootstrap CI (Rubin 1981, Dirichlet(1,...,1) weights).
- Pipeline sampling pass (temperature > 0, `n_samples > 0`) wired through duckdb's new `judge_uq_results` table.
- Rubric templates for code, math, factuality, and free-form quality.

**M3 — routing + aggregation + decomposition**

- **Hierarchical-Gaussian jury aggregator** for continuous scores — closed-form EM for `score_ij = θ_i + b_j + ε_ij`, recovers per-item posteriors and per-judge bias/precision with the standard hierarchical M-step correction for σ.
- **Aleatoric/epistemic variance decomposition** (Kendall & Gal 2017; Depeweg et al. 2018) via nested bootstrap — outer over judges (epistemic), inner over temperature samples (aleatoric).
- **`JuryRouter` Protocol** + four strategies: `all`, `single`, `escalation` (cheap-first, escalate on high inter-judge variance), and `bandit` (Thompson sampling on Beta(α,β) per `(judge, task_family)`, reward = info-per-dollar vs median).
- CLI flags: `--strategy {all,single,escalation,bandit}`, `--single-judge`, `--escalation-tau`, `--bandit-top-k`, `--bandit-seed`. `--uq decomposition` triggers the aleatoric/epistemic computation.

**M4 — statistics + dashboard**

- `stats/coverage_tests.py` — marginal coverage with Clopper-Pearson CI, conditional-per-group with Bonferroni p-values, Hosmer-Lemeshow binning.
- `stats/reliability.py` — ECE, MCE, Brier (Naeini et al. 2015, Guo et al. 2017), reliability diagram with 95% bootstrap bands (Bröcker & Smith 2007).
- `stats/compare.py` — paired-bootstrap Spearman ρ / Kendall τ between judges; permutation test for "judges A and B disagree more than chance" (Pitman 1937).
- `stats/pareto.py` — coverage-width Pareto sweep over α.
- `stats/bootstrap.py` — pivot CI, paired-difference bootstrap, Bayesian (Dirichlet-weights) bootstrap.
- `panoptes report --db <duckdb> --out report.html` — self-contained offline HTML with run metadata, cost-by-judge, UQ-result counts, inter-judge agreement.
- **Streamlit dashboard** — `uv run streamlit run src/panoptes/dashboard/app.py -- --db ...`. Pages: Overview, Drill-down, Judge comparison, Conformal Pareto. Reads duckdb directly via `st.cache_data`; 1k-row design target.

**M5 — benchmarks + sandbox + polish**

- Benchmark loaders: **MBPP**, **GSM8K** (with parsed `#### N` final answer), **MT-Bench**, **TruthfulQA** (parquet from HF). All cached via `http_fetch_cached`.
- `sandbox/python_exec.py` — subprocess Python sandbox with `resource.setrlimit` CPU+memory caps and hard wall timeout. `humaneval_check(prompt, candidate, test, entry_point)` convenience for HumanEval grading.
- **Calibration probe** — `benchmarks/calibration_probe.py` mechanically obfuscates HumanEval problems by renaming entry-point functions and rewriting tests so judges can't recall the canonical solution by name. Graded via the sandbox for an after-pretraining boolean label per item.
- `examples/custom_judge.py` + `examples/custom_provider.py` — single-file walkthroughs of the M5 acceptance criterion: new judge or provider in one Protocol class, no other code changes.
- Nightly real-provider CI (`.github/workflows/nightly.yml`) gated on secrets.
- CI coverage gate `--cov-fail-under=80` on `panoptes.uq` + `panoptes.stats`.

## Install

```sh
uv sync --extra dev
```

Set provider keys in a `.env` next to `pyproject.toml`:

```sh
ANTHROPIC_API_KEY=sk-ant-...
```

## Quick start

Run a smoke with the deterministic mock client (no API key needed):

```sh
# M1 path
uv run panoptes eval humaneval --judges claude --uq split --n 5 --mock --out runs/smoke.duckdb

# M2 path — multi-judge, sampling-based UQ, LLM-as-NLI semantic entropy
uv run panoptes eval humaneval \
  --judges claude,gpt,gemini \
  --uq split,self-consistency,semantic-entropy \
  --n 5 --n-samples 5 --nli llm --mock \
  --out runs/m2_smoke.duckdb
```

With real provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`):

```sh
uv run panoptes eval humaneval \
  --judges claude,gpt,gemini \
  --uq split,self-consistency,semantic-entropy \
  --n 20 --n-samples 10 \
  --out runs/v2.duckdb
```

## Results

> TODO: measure. v1.0 acceptance criteria target marginal coverage within 2pp of nominal on ≥3 of 4 conformal methods, and a 30%+ bandit-vs-all-judges cost reduction on cost-per-correctly-flagged-uncertain example. M1 ships the wiring; the headline numbers land in M2–M5.

## Roadmap

| Milestone | Scope |
|---|---|
| **M1** ✅ | Foundation: schemas, Anthropic client, rubric judge, split conformal, HumanEval, DuckDB, CLI, CI. |
| **M2** ✅ | UQ breadth: OpenAI / Google / OpenAI-compat clients; adaptive (CQR) and Mondrian conformal; self-consistency; semantic entropy (Farquhar et al. 2024). |
| **M3** ✅ | Routing: hierarchical-Gaussian jury aggregation; aleatoric/epistemic decomposition; escalation + Thompson-sampling bandit. |
| **M4** ✅ | Statistics & dashboard: coverage diagnostics, reliability diagrams with bootstrap bands, coverage-width Pareto, Streamlit dashboard over DuckDB. |
| **M5** ✅ | Remaining benchmarks (MBPP, GSM8K, TruthfulQA + BM25, MT-Bench), sandboxed Python execution, calibration probe, polish. |

See `METHODS.md` for citations and math, and `CONTRIBUTING.md` for the design principles.

## Development

```sh
uv run ruff check .
uv run pyright
uv run pytest -q
```

PANOPTES is **Python-only** (uv-managed). No Node toolchain anywhere — including the dashboard, which is Streamlit (M4). No `npm install` step in build, dev, or CI.

## License

MIT (see `LICENSE` once added).
