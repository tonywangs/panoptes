# Contributing to PANOPTES

## Design principles

1. **Statistics first.** Every reported number has a confidence interval. Bootstrap with explicit seeds. No bare point estimates.
2. **Aleatoric vs epistemic, always separated.** `Var_total = E[Var(score|judge)] + Var(E[score|judge])`. Both halves are first-class.
3. **Provider-agnostic, async-everywhere.** One `LLMClient` Protocol. Per-provider `asyncio.Semaphore` for rate limits. No sync calls in hot paths.
4. **Strict typing as contract.** `pyright --strict` is a build gate. All public surfaces are Pydantic v2 models. No `Any` leakage.
5. **Determinism where the API allows.** Seed `numpy` and `random`. Be honest about provider non-determinism: even `temperature=0` is best-effort.
6. **Fail loud on terminal errors.** 4xx auth/validation surfaces immediately. Only 429/5xx/timeout/overloaded retry — with bounded retries, exponential backoff, and full jitter.
7. **Cache-aware costs.** Anthropic prompt-cache markers on rubric/system content. Cost reports break down `cache_read` / `cache_write` / fresh tokens.
8. **No emojis** in code, logs, or commits.
9. **No invented numbers** in README. Unmeasured results are `TODO: measure`.
10. **No npm / no Node.** Stack is Python-only (uv-managed). The dashboard is Streamlit; no JS toolchain anywhere.

## Adding a provider

The pipeline only talks to `panoptes.clients.base.LLMClient`. To add a new provider:

1. Write a class implementing the `LLMClient` Protocol. See `src/panoptes/clients/anthropic.py` as a reference.
2. Map provider-specific error shapes onto `RetriableError` / `TerminalError` from `panoptes.errors`.
3. Register the model's pricing via `panoptes.clients.base.register_pricing(model, ModelPricing(...))`.
4. Add a judge alias in `cli.py`'s `_JUDGE_ALIASES` so the CLI can resolve it.

No other code changes required. The example in `examples/custom_provider.py` (lands in M5) walks through the full path.

## Adding a UQ method

Implement a class with `fit(...)` and a `predict_*(...)` method in `src/panoptes/uq/`. Cite the paper in the module docstring. Add:

- Unit tests in `tests/unit/uq/` with at least one synthetic-data coverage / sanity check.
- A property test in `tests/property/` if the method has an obvious invariant (monotonicity, bounds-preservation, etc.).
- A section in `METHODS.md` with the math and citations.

## Adding a benchmark

Implement a loader in `src/panoptes/benchmarks/` that:

1. Fetches data via `panoptes.benchmarks.loader.http_fetch_cached(...)` or HF `datasets`.
2. Returns `list[BenchmarkItem]` with the appropriate `task_family`.
3. Stores benchmark-specific fields (executable tests, reference answers, etc.) under `BenchmarkItem.metadata`.

## Development workflow

```sh
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
```

The CI mirrors this exactly. The nightly job (M5) additionally exercises real providers via `pytest -m real_provider`.
