# PANOPTES

**Uncertainty-aware LLM evaluation.** Calibrated posteriors. Aleatoric/epistemic decomposition. Conformal prediction.

PANOPTES treats LLM evaluation as a first-class statistical inference problem rather than a "score-and-stop" exercise. For every `(task, response, judge)` tuple, the framework asks: what is the posterior distribution over the true quality, and what fraction of that uncertainty is reducible (epistemic) vs irreducible (aleatoric)? That decomposition tells you whether to trust the score, sample more judges, escalate to a stronger one, or accept the response is genuinely ambiguous.

> Status: **M1 (foundation) shipped.** Full v1.0 roadmap below; numbers under "Results" are deliberately empty until measured.

## What's in M1

- Async, provider-agnostic `LLMClient` Protocol with an Anthropic Messages-API impl (httpx, prompt-cache aware, exponential backoff with full jitter, retriable vs terminal error taxonomy).
- Pydantic v2 schemas for every cross-module contract: `EvalRecord`, `JudgeResponse`, `RubricScore`, `ConformalResult`, `CostReport`, etc. `pyright --strict` clean.
- Single-call rubric judge with versioned, content-hashed prompt templates.
- **Split conformal prediction** (Vovk/Gammerman/Shafer 2005; Papadopoulos et al. 2002) over bounded scores, with the finite-sample `ceil((n+1)(1-α))/n` quantile correction.
- HumanEval loader (content-hashed HTTP cache) + DuckDB result store partitioned by `(task_family, judge_id, prompt_version_hash)`.
- Typer CLI: `panoptes eval humaneval --judges claude --uq split --n 5`.
- Mocked-client smoke path + property tests with `hypothesis` + respx-based provider tests.

## Install

```sh
uv sync --extra dev
```

Set provider keys in a `.env` next to `pyproject.toml`:

```sh
ANTHROPIC_API_KEY=sk-ant-...
```

## Quick start

Run the M1 smoke without an API key (deterministic mock judge):

```sh
uv run panoptes eval humaneval --judges claude --uq split --n 5 --mock --out runs/smoke.duckdb
```

With a real Anthropic key:

```sh
uv run panoptes eval humaneval --judges claude --uq split --n 5 --out runs/v1.duckdb
```

## Results

> TODO: measure. v1.0 acceptance criteria target marginal coverage within 2pp of nominal on ≥3 of 4 conformal methods, and a 30%+ bandit-vs-all-judges cost reduction on cost-per-correctly-flagged-uncertain example. M1 ships the wiring; the headline numbers land in M2–M5.

## Roadmap

| Milestone | Scope |
|---|---|
| **M1** ✅ | Foundation: schemas, Anthropic client, rubric judge, split conformal, HumanEval, DuckDB, CLI, CI. |
| **M2** | UQ breadth: OpenAI / Google / OpenAI-compat clients; adaptive (CQR) and Mondrian conformal; self-consistency; semantic entropy (Farquhar et al. 2024). |
| **M3** | Routing: hierarchical-Gaussian jury aggregation; aleatoric/epistemic decomposition; escalation + Thompson-sampling bandit. |
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
