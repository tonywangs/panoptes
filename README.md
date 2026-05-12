# PANOPTES

**Uncertainty-aware LLM evaluation.** Calibrated posteriors. Aleatoric/epistemic decomposition. Conformal prediction.

PANOPTES treats LLM evaluation as a first-class statistical inference problem rather than a "score-and-stop" exercise. For every `(task, response, judge)` tuple, the framework asks: what is the posterior distribution over the true quality, and what fraction of that uncertainty is reducible (epistemic) vs irreducible (aleatoric)? That decomposition tells you whether to trust the score, sample more judges, escalate to a stronger one, or accept the response is genuinely ambiguous.

> Status: **M1 (foundation) shipped.** Full v1.0 roadmap below; numbers under "Results" are deliberately empty until measured.

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
| **M4** | Statistics & dashboard: coverage diagnostics, reliability diagrams with bootstrap bands, coverage-width Pareto, Streamlit dashboard over DuckDB. |
| **M5** | Remaining benchmarks (MBPP, GSM8K, TruthfulQA + BM25, MT-Bench), sandboxed Python execution, calibration probe, polish. |

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
