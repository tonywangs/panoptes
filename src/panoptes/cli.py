"""`panoptes` CLI entrypoint.

Subcommands:
    eval BENCHMARK [...]   run an evaluation
    calibrate              (M4)
    report                 (M4)
    route                  (M3)

The CLI is intentionally thin: it parses arguments, builds the right objects,
and calls `run_evaluation`. All the interesting logic lives in `pipeline.py`
and the modules it composes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from panoptes._version import __version__
from panoptes.benchmarks.humaneval import load_humaneval
from panoptes.clients._mock import MockClient
from panoptes.clients.anthropic import AnthropicClient
from panoptes.clients.base import LLMClient
from panoptes.config import load_settings
from panoptes.errors import ConfigError
from panoptes.judges.base import load_prompt_template
from panoptes.judges.rubric import RubricJudge
from panoptes.pipeline import EvalConfig, JudgeRef, new_run_id, run_evaluation
from panoptes.schemas import BenchmarkItem, CostReport
from panoptes.storage.duckdb_store import DuckDBStore
from panoptes.storage.prompt_cache import PromptCache

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


# Short judge aliases acceptable on the CLI -> (provider, default model, prompt path).
_JUDGE_ALIASES: dict[str, tuple[str, str, str]] = {
    "claude": ("anthropic", "claude-sonnet-4-6", "prompts/rubric_code_v1.md"),
    "claude-haiku": ("anthropic", "claude-haiku-4-5", "prompts/rubric_code_v1.md"),
    "claude-opus": ("anthropic", "claude-opus-4-7", "prompts/rubric_code_v1.md"),
}


def _resolve_judge(alias: str, *, mock: bool, project_root: Path) -> JudgeRef:
    """Build a `JudgeRef` from a CLI judge alias.

    For M1, only Anthropic-backed aliases resolve. Future milestones extend
    `_JUDGE_ALIASES` and the provider dispatch. If `mock=True`, we substitute
    a `MockClient` so the CLI can run without API keys.
    """
    if alias not in _JUDGE_ALIASES:
        raise typer.BadParameter(f"Unknown judge alias '{alias}'. Known: {sorted(_JUDGE_ALIASES)}")
    provider, model, prompt_rel = _JUDGE_ALIASES[alias]
    if provider != "anthropic":
        raise typer.BadParameter(f"Provider '{provider}' is not yet supported in M1.")
    template_path = project_root / prompt_rel
    if not template_path.exists():
        raise typer.BadParameter(
            f"Prompt template not found at {template_path}. "
            "Run `panoptes` from the repository root, or symlink prompts/."
        )
    template = load_prompt_template(template_path)
    client: LLMClient
    if mock:
        client = MockClient(provider="anthropic", model=model)
    else:
        settings = load_settings()
        try:
            api_key = settings.api_key("ANTHROPIC_API_KEY")
        except ConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
        client = AnthropicClient(
            api_key=api_key,
            model=model,
            rate_limit=settings.rate_limits.get(provider),
            request_timeout_s=settings.request_timeout_s,
            connect_timeout_s=settings.connect_timeout_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            backoff_max_s=settings.backoff_max_s,
        )
    variant = template_path.stem  # e.g. 'rubric_code_v1'
    judge = RubricJudge(client=client, template=template, variant=variant)
    return JudgeRef(judge=judge, prompt_version_hash=template.content_hash)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed PANOPTES version."""
    console.print(f"panoptes {__version__}")


@app.command(name="eval")
def eval_cmd(
    benchmark: Annotated[str, typer.Argument(help="Benchmark name (M1: humaneval).")],
    judges: Annotated[
        str,
        typer.Option(
            "--judges",
            help="Comma-separated judge aliases (e.g. 'claude' or 'claude,claude-haiku').",
        ),
    ] = "claude",
    uq: Annotated[
        str, typer.Option("--uq", help="Comma-separated UQ methods (M1: 'split').")
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
    config = EvalConfig(
        run_id=new_run_id(),
        alpha=alpha,
        uq_methods=uq_methods,
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
            )
        )
        _print_summary(cost, out=out, n_items=len(items), run_id=config.run_id)


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
    raise typer.BadParameter(f"Unknown benchmark '{name}'. M1 supports: humaneval.")


def _build_responses(benchmark: str, items: list[BenchmarkItem]) -> dict[str, str]:
    """For M1: synthesize 'model responses' from each item's canonical solution.

    This is a placeholder so the smoke pipeline has something to judge before
    M5 wires up an actual response-generation step. In practice, callers
    should supply pre-generated responses via a (future) `--responses` flag.
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


def _print_summary(cost: CostReport, *, out: Path, n_items: int, run_id: str) -> None:
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
