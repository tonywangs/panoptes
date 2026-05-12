"""`panoptes` CLI entrypoint.

Subcommands:
    eval BENCHMARK [...]   run an evaluation
    report                 render an offline HTML report from a duckdb file
    version                print the installed version

The CLI is intentionally thin: it parses arguments, builds the right objects,
and calls `run_evaluation`. All the interesting logic lives in `pipeline.py`
and the modules it composes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from panoptes._version import __version__
from panoptes.benchmarks.humaneval import load_humaneval
from panoptes.clients._mock import MockClient
from panoptes.clients.anthropic import AnthropicClient
from panoptes.clients.base import LLMClient
from panoptes.clients.google import GoogleClient
from panoptes.clients.openai import OpenAIClient
from panoptes.clients.openai_compat import GroqClient, TogetherClient
from panoptes.config import Settings, load_settings
from panoptes.errors import ConfigError
from panoptes.judges.base import load_prompt_template
from panoptes.judges.rubric import RubricJudge
from panoptes.pipeline import EvalConfig, JudgeRef, new_run_id, run_evaluation
from panoptes.routing.bandit import ThompsonBandit
from panoptes.routing.base import JuryRouter
from panoptes.routing.strategies import AllJudges, EscalationPolicy, SingleJudge
from panoptes.schemas import BenchmarkItem, CostReport
from panoptes.storage.duckdb_store import DuckDBStore
from panoptes.storage.prompt_cache import PromptCache
from panoptes.uq.nli.base import NLIBackend
from panoptes.uq.nli.llm import LLMNLIBackend

app = typer.Typer(
    name="panoptes",
    help="Uncertainty-aware LLM evaluation. See README.md.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# ---------------------------------------------------------------------------
# Judge spec resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _JudgeAlias:
    """Resolved metadata for a CLI judge alias."""

    provider: str
    model: str
    prompt_rel: str
    env_var: str
    cost_tier: str = "mid"  # "cheap" | "mid" | "expensive" — read by the router


_JUDGE_ALIASES: dict[str, _JudgeAlias] = {
    # Anthropic
    "claude": _JudgeAlias(
        "anthropic", "claude-sonnet-4-6", "prompts/rubric_code_v1.md", "ANTHROPIC_API_KEY", "mid"
    ),
    "claude-haiku": _JudgeAlias(
        "anthropic", "claude-haiku-4-5", "prompts/rubric_code_v1.md", "ANTHROPIC_API_KEY", "cheap"
    ),
    "claude-opus": _JudgeAlias(
        "anthropic",
        "claude-opus-4-7",
        "prompts/rubric_code_v1.md",
        "ANTHROPIC_API_KEY",
        "expensive",
    ),
    # OpenAI
    "gpt": _JudgeAlias(
        "openai", "gpt-4o", "prompts/rubric_code_v1.md", "OPENAI_API_KEY", "mid"
    ),
    "gpt-mini": _JudgeAlias(
        "openai", "gpt-4o-mini", "prompts/rubric_code_v1.md", "OPENAI_API_KEY", "cheap"
    ),
    # Google
    "gemini": _JudgeAlias(
        "google", "gemini-2.5-pro", "prompts/rubric_code_v1.md", "GOOGLE_API_KEY", "mid"
    ),
    # OpenAI-compat
    "together-llama": _JudgeAlias(
        "together",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "prompts/rubric_code_v1.md",
        "TOGETHER_API_KEY",
        "mid",
    ),
    "groq-llama": _JudgeAlias(
        "groq",
        "llama-3.1-70b-versatile",
        "prompts/rubric_code_v1.md",
        "GROQ_API_KEY",
        "cheap",
    ),
}


def _build_client(spec: _JudgeAlias, *, mock: bool, settings: Settings) -> LLMClient:
    """Instantiate the right `LLMClient` for `spec.provider`."""
    if mock:
        return MockClient(provider=spec.provider, model=spec.model)
    try:
        api_key = settings.api_key(spec.env_var)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    rate_limit = settings.rate_limits.get(spec.provider)
    if spec.provider == "anthropic":
        return AnthropicClient(
            api_key=api_key,
            model=spec.model,
            rate_limit=rate_limit,
            request_timeout_s=settings.request_timeout_s,
            connect_timeout_s=settings.connect_timeout_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            backoff_max_s=settings.backoff_max_s,
        )
    if spec.provider == "openai":
        return OpenAIClient(
            api_key=api_key,
            model=spec.model,
            rate_limit=rate_limit,
            request_timeout_s=settings.request_timeout_s,
            connect_timeout_s=settings.connect_timeout_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            backoff_max_s=settings.backoff_max_s,
        )
    if spec.provider == "google":
        return GoogleClient(
            api_key=api_key,
            model=spec.model,
            rate_limit=rate_limit,
            request_timeout_s=settings.request_timeout_s,
            connect_timeout_s=settings.connect_timeout_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            backoff_max_s=settings.backoff_max_s,
        )
    if spec.provider == "together":
        return TogetherClient(
            api_key=api_key,
            model=spec.model,
            rate_limit=rate_limit,
            request_timeout_s=settings.request_timeout_s,
            connect_timeout_s=settings.connect_timeout_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            backoff_max_s=settings.backoff_max_s,
        )
    if spec.provider == "groq":
        return GroqClient(
            api_key=api_key,
            model=spec.model,
            rate_limit=rate_limit,
            request_timeout_s=settings.request_timeout_s,
            connect_timeout_s=settings.connect_timeout_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            backoff_max_s=settings.backoff_max_s,
        )
    raise typer.BadParameter(f"Unsupported provider '{spec.provider}'.")


def _resolve_judge(alias: str, *, mock: bool, project_root: Path) -> JudgeRef:
    """Build a `JudgeRef` from a CLI judge alias.

    `mock=True` substitutes a `MockClient` so the CLI can run without keys.
    """
    if alias not in _JUDGE_ALIASES:
        raise typer.BadParameter(
            f"Unknown judge alias '{alias}'. Known: {sorted(_JUDGE_ALIASES)}"
        )
    spec = _JUDGE_ALIASES[alias]
    template_path = project_root / spec.prompt_rel
    if not template_path.exists():
        raise typer.BadParameter(
            f"Prompt template not found at {template_path}. "
            "Run `panoptes` from the repository root, or symlink prompts/."
        )
    template = load_prompt_template(template_path)
    settings = load_settings()
    client = _build_client(spec, mock=mock, settings=settings)
    variant = template_path.stem  # e.g. 'rubric_code_v1'
    judge = RubricJudge(client=client, template=template, variant=variant)
    tier = spec.cost_tier if spec.cost_tier in {"cheap", "mid", "expensive"} else "mid"
    return JudgeRef(
        judge=judge,
        prompt_version_hash=template.content_hash,
        cost_tier=tier,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed PANOPTES version."""
    console.print(f"panoptes {__version__}")


@app.command(name="report")
def report_cmd(
    db: Annotated[Path, typer.Option("--db", help="PANOPTES duckdb file.")],
    out: Annotated[
        Path, typer.Option("--out", help="Output HTML path.")
    ] = Path("report.html"),
) -> None:
    """Render an offline HTML report from a duckdb file."""
    from panoptes.reporting import write_report  # noqa: PLC0415

    if not db.exists():
        raise typer.BadParameter(f"duckdb file not found: {db}")
    write_report(db, out)
    console.print(f"wrote report to [bold]{out}[/bold]")


@app.command(name="eval")
def eval_cmd(
    benchmark: Annotated[
        str,
        typer.Argument(
            help="Benchmark name: humaneval | mbpp | gsm8k | mtbench | truthfulqa."
        ),
    ],
    judges: Annotated[
        str,
        typer.Option(
            "--judges",
            help="Comma-separated judge aliases (e.g. 'claude' or 'claude,claude-haiku').",
        ),
    ] = "claude",
    uq: Annotated[
        str,
        typer.Option(
            "--uq",
            help="Comma-separated UQ methods: split, adaptive, mondrian, self-consistency, semantic-entropy.",
        ),
    ] = "split",
    n: Annotated[int, typer.Option("--n", help="Truncate benchmark to first N items.")] = 5,
    alpha: Annotated[float, typer.Option("--alpha", help="Conformal miscoverage rate.")] = 0.1,
    out: Annotated[Path, typer.Option("--out", help="DuckDB result file path.")] = Path(
        "runs/panoptes.duckdb"
    ),
    mock: Annotated[
        bool,
        typer.Option("--mock", help="Use a deterministic in-memory client. No API key needed."),
    ] = False,
    n_samples: Annotated[
        int,
        typer.Option(
            "--n-samples",
            help="Sampling-pass MC draws per (judge, item). 0 = auto (10 if needed).",
        ),
    ] = 0,
    temperature_sampling: Annotated[
        float,
        typer.Option(
            "--temperature-sampling",
            help="Temperature for the sampling pass. Ignored when n-samples == 0.",
        ),
    ] = 1.0,
    nli: Annotated[
        str,
        typer.Option(
            "--nli",
            help="NLI backend for semantic-entropy: 'llm' (default) or 'deberta'.",
        ),
    ] = "llm",
    nli_judge: Annotated[
        str,
        typer.Option(
            "--nli-judge",
            help="When --nli=llm, judge alias used to classify entailment.",
        ),
    ] = "claude-haiku",
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help="Router strategy: 'all' (default), 'single', 'escalation', or 'bandit'.",
        ),
    ] = "all",
    single_judge: Annotated[
        str | None,
        typer.Option(
            "--single-judge",
            help="Judge alias for --strategy=single (else: cheapest tier in the judge list).",
        ),
    ] = None,
    escalation_tau: Annotated[
        float,
        typer.Option(
            "--escalation-tau",
            help="Inter-judge variance threshold above which --strategy=escalation calls an expensive judge.",
        ),
    ] = 0.02,
    bandit_top_k: Annotated[
        int,
        typer.Option(
            "--bandit-top-k",
            help="Number of judges to Thompson-sample per item under --strategy=bandit.",
        ),
    ] = 2,
    bandit_seed: Annotated[
        int,
        typer.Option("--bandit-seed", help="Seed for the Thompson-sampling RNG."),
    ] = 0,
    model_under_test: Annotated[
        str,
        typer.Option(
            "--model-under-test",
            help="Label for the model whose responses are being judged.",
        ),
    ] = "reference_solution",
    log_level: Annotated[str, typer.Option("--log-level", help="Python logging level.")] = "INFO",
) -> None:
    """Run an evaluation end-to-end and write rows to `--out`."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    project_root = _project_root()
    settings = load_settings()
    items = _load_benchmark(benchmark, n=n, cache_dir=settings.cache_dir)
    responses = _build_responses(benchmark, items)
    judge_refs = [
        _resolve_judge(j.strip(), mock=mock, project_root=project_root) for j in judges.split(",")
    ]
    uq_methods = tuple(m.strip() for m in uq.split(",") if m.strip())
    if strategy not in {"all", "single", "escalation", "bandit"}:
        raise typer.BadParameter(
            f"--strategy={strategy!r} not in 'all'/'single'/'escalation'/'bandit'."
        )
    strat: Literal["all", "single", "escalation", "bandit"] = strategy  # type: ignore[assignment]
    config = EvalConfig(
        run_id=new_run_id(),
        alpha=alpha,
        uq_methods=uq_methods,
        n_samples=n_samples,
        temperature_sampling=temperature_sampling,
        strategy=strat,
    )
    router = _build_router(
        strategy=strat,
        single_judge=single_judge,
        escalation_tau=escalation_tau,
        bandit_top_k=bandit_top_k,
        bandit_seed=bandit_seed,
        judge_refs=judge_refs,
    )
    nli_backend = _resolve_nli_backend(
        nli, nli_judge=nli_judge, mock=mock, project_root=project_root, config=config
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with DuckDBStore.open(out) as store:
        cache = PromptCache(store=store)
        cost = asyncio.run(
            run_evaluation(
                items=items,
                responses=responses,
                judges=judge_refs,
                store=store,
                cache=cache,
                config=config,
                model_under_test=model_under_test,
                nli_backend=nli_backend,
                router=router,
            )
        )
        _print_summary(
            cost,
            out=out,
            n_items=len(items),
            run_id=config.run_id,
            n_uq_results=store.count_uq_results(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Find the PANOPTES project root by walking up from CWD looking for `prompts/`.

    Falls back to CWD if not found, which lets users run from anywhere
    provided they pass `--prompt-root` (future flag) or symlink prompts/.
    """
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "prompts").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return here


def _load_benchmark(name: str, *, n: int, cache_dir: Path) -> list[BenchmarkItem]:
    if name == "humaneval":
        return load_humaneval(cache_dir=cache_dir, limit=n)
    raise typer.BadParameter(
        f"Unknown benchmark '{name}'. CLI currently loads: humaneval."
    )


def _build_responses(benchmark: str, items: list[BenchmarkItem]) -> dict[str, str]:
    """Synthesize 'model responses' from each item's canonical solution.

    This is a placeholder so the smoke pipeline has something to judge until
    callers supply pre-generated responses via a `--responses` flag (planned
    follow-up). For real grading runs, generate responses ahead of time and
    feed them through `run_evaluation` directly.
    """
    if benchmark != "humaneval":
        return {}
    responses: dict[str, str] = {}
    for item in items:
        canonical = item.metadata.get("canonical_solution")
        if isinstance(canonical, str):
            # Prepend the function signature so the response is self-contained.
            responses[item.item_id] = item.prompt + canonical
    return responses


def _build_router(
    *,
    strategy: Literal["all", "single", "escalation", "bandit"],
    single_judge: str | None,
    escalation_tau: float,
    bandit_top_k: int,
    bandit_seed: int,
    judge_refs: list[JudgeRef],
) -> JuryRouter:
    """Instantiate the routing strategy from CLI flags."""
    if strategy == "all":
        return AllJudges()
    if strategy == "single":
        if single_judge is not None:
            valid = {ref.judge.judge_id for ref in judge_refs}
            # Match either the bare alias or the full provider:model:variant id.
            chosen = single_judge
            if chosen not in valid:
                matches = [jid for jid in valid if jid.endswith(f":{chosen}") or chosen in jid]
                if len(matches) == 1:
                    chosen = matches[0]
                else:
                    raise typer.BadParameter(
                        f"--single-judge={single_judge!r} did not match any of {sorted(valid)}"
                    )
            return SingleJudge(judge_id=chosen)
        return SingleJudge()
    if strategy == "escalation":
        return EscalationPolicy(tau=escalation_tau)
    if strategy == "bandit":
        return ThompsonBandit(top_k=bandit_top_k, seed=bandit_seed)
    raise typer.BadParameter(f"Unknown strategy '{strategy}'")


def _resolve_nli_backend(
    backend: str,
    *,
    nli_judge: str,
    mock: bool,
    project_root: Path,
    config: EvalConfig,
) -> NLIBackend | None:
    """Build the NLI backend if any selected UQ method needs it; else None.

    `deberta` requires the `providers-hf` extra (transformers + torch).
    `llm` reuses the CLI's judge resolution so the NLI judge can be any
    alias (defaulting to a cheap Haiku for cost reasons).
    """
    if "semantic-entropy" not in config.uq_methods:
        return None
    if backend == "llm":
        if nli_judge not in _JUDGE_ALIASES:
            raise typer.BadParameter(
                f"--nli-judge='{nli_judge}' not in {sorted(_JUDGE_ALIASES)}"
            )
        spec = _JUDGE_ALIASES[nli_judge]
        settings = load_settings()
        client = _build_client(spec, mock=mock, settings=settings)
        return LLMNLIBackend(client=client)
    if backend == "deberta":
        try:
            from panoptes.uq.nli.deberta import DebertaNLIBackend  # noqa: PLC0415
        except ImportError as exc:
            raise typer.BadParameter(
                "--nli=deberta requires `transformers` and `torch`. "
                "Install via: uv sync --extra providers-hf"
            ) from exc
        return DebertaNLIBackend()
    raise typer.BadParameter(f"Unknown NLI backend '{backend}'. Try 'llm' or 'deberta'.")


def _print_summary(
    cost: CostReport,
    *,
    out: Path,
    n_items: int,
    run_id: str,
    n_uq_results: int,
) -> None:
    table = Table(title=f"PANOPTES run {run_id}", show_lines=False)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    table.add_row("items", str(n_items))
    table.add_row("judge calls", str(cost.n_calls))
    table.add_row("input tokens", f"{cost.input_tokens:,}")
    table.add_row("output tokens", f"{cost.output_tokens:,}")
    table.add_row("cache_read tokens", f"{cost.cache_read_tokens:,}")
    table.add_row("cache_creation tokens", f"{cost.cache_creation_tokens:,}")
    table.add_row("usd_total", f"${cost.usd_total:.6f}")
    table.add_row("uq results", str(n_uq_results))
    table.add_row("output", str(out))
    console.print(table)
    if cost.by_judge:
        by = Table(title="Cost by judge", show_lines=False)
        by.add_column("judge_id", style="cyan")
        by.add_column("usd", style="white")
        for judge_id, usd in sorted(cost.by_judge.items()):
            by.add_row(judge_id, f"${usd:.6f}")
        console.print(by)


if __name__ == "__main__":
    app()
