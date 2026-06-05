"""End-to-end held-out calibration probe with real ground-truth labels.

For each obfuscated HumanEval item:
    1. A *weaker* LLM (gpt-4o-mini at temperature 0.7) writes a candidate
       solution. The obfuscated entry-point name prevents memorization, so
       the model has to actually solve the task.
    2. The candidate runs in the sandbox against the rewritten tests. The
       boolean pass/fail is our ground-truth label.
    3. Two judges (Claude Sonnet + GPT-4o) score the (problem, candidate)
       pair on the standard [0, 1] rubric.
    4. We split the items 50/50 into a calibration set and a held-out test
       set, fit split conformal on the calibration residuals
       `|judge_score − is_correct|`, and measure empirical coverage on the
       test set. Compare to the nominal `1 − α`.

This is the headline statistical-validity claim PANOPTES is built around:
empirical coverage on real LLM-judged data should land within a few
percentage points of nominal. Numbers print to console for screenshot.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from panoptes.benchmarks.calibration_probe import obfuscate_humaneval
from panoptes.benchmarks.humaneval import load_humaneval
from panoptes.clients.anthropic import AnthropicClient
from panoptes.clients.base import Message
from panoptes.clients.openai import OpenAIClient
from panoptes.config import load_settings
from panoptes.judges.base import load_prompt_template
from panoptes.judges.rubric import RubricJudge
from panoptes.sandbox.python_exec import humaneval_check
from panoptes.uq.conformal_split import split_conformal_quantile

console = Console()

N_ITEMS = 50
GEN_MODEL = "gpt-4o-mini"
GEN_TEMPERATURE = 0.7   # mid-temp so we get a *mix* of pass/fail candidates
JUDGES = [
    ("anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    ("openai", "gpt-4o", "OPENAI_API_KEY"),
]
ALPHAS = [0.05, 0.1, 0.2, 0.3]
CAL_FRACTION = 0.5
SEED = 0


_FENCE_RE = re.compile(r"^```(?:python)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """LLMs often wrap code in ```python ... ``` even when told not to.
    Strip the fences and return only the code inside; pass through otherwise."""
    match = _FENCE_RE.search(text.strip())
    if match:
        return match.group(1)
    return text


def _extract_body(prompt: str, full_text: str, entry_point: str) -> str:
    """The LLM ideally returns just the function body, indented under the
    signature. If it returns the whole function (def header + body), we
    strip the header line(s) up to and including the signature and keep the
    body. This is brittle for unusual LLM outputs, but for short HumanEval
    functions it works in practice. Failures here will be caught by the
    sandbox (the candidate will throw NameError / IndentationError) and
    surface as `is_correct = False`, which is the right outcome."""
    text = _strip_markdown(full_text)
    if f"def {entry_point}" in text:
        # Find the def line, then keep everything after it.
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith(f"def {entry_point}"):
                return "\n".join(lines[i + 1 :])
    return text


async def generate_candidates(
    probes: list, settings, openai_key: str
) -> list[str]:
    client = OpenAIClient(api_key=openai_key, model=GEN_MODEL, max_retries=3)
    sys = (
        "You are a careful Python programmer. Read the function signature and "
        "docstring, then write the function body. Reply with ONLY the body "
        "indented appropriately under the signature — no markdown, no "
        "explanation, just code that can be appended directly under the def line."
    )
    try:
        from panoptes.clients.base import SystemBlock

        async def gen_one(probe) -> str:
            try:
                resp = await client.complete(
                    messages=[Message(role="user", content=probe.item.prompt)],
                    system=[SystemBlock(text=sys, cache_control="ephemeral")],
                    max_tokens=512,
                    temperature=GEN_TEMPERATURE,
                )
                return _extract_body(
                    probe.item.prompt, resp.text, probe.rewritten_entry_point
                )
            except Exception as exc:
                console.print(f"[red]generation failed for {probe.item.item_id}: {exc}[/]")
                return ""

        return await asyncio.gather(*[gen_one(p) for p in probes])
    finally:
        await client.aclose()


def grade(probes: list, candidates: list[str]) -> list[float]:
    labels: list[float] = []
    for probe, body in zip(probes, candidates, strict=True):
        if not body.strip():
            labels.append(0.0)
            continue
        result = humaneval_check(
            prompt=probe.item.prompt,
            candidate=body,
            test=probe.rewritten_test,
            entry_point=probe.rewritten_entry_point,
        )
        labels.append(1.0 if result.passed else 0.0)
    return labels


async def judge_candidates(
    probes: list,
    candidates: list[str],
    judge: RubricJudge,
) -> list[float]:
    async def score_one(probe, body: str) -> float:
        full_response = probe.item.prompt + body
        try:
            jr = await judge.evaluate(probe.item, full_response)
            return jr.score.value
        except Exception as exc:
            console.print(
                f"[yellow]judge {judge.judge_id} failed on {probe.item.item_id}: {exc}[/]"
            )
            return float("nan")

    return await asyncio.gather(
        *[score_one(p, b) for p, b in zip(probes, candidates, strict=True)]
    )


def fit_and_evaluate(
    judge_scores: dict[str, list[float]],
    labels: list[float],
    *,
    seed: int,
) -> list[dict]:
    """For each judge × alpha, fit conformal on the calibration split and
    measure empirical coverage on the held-out test split."""
    labels_arr = np.asarray(labels, dtype=np.float64)
    n = len(labels_arr)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = int(n * CAL_FRACTION)
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]
    rows: list[dict] = []
    for judge_id, scores_list in judge_scores.items():
        scores = np.asarray(scores_list, dtype=np.float64)
        # Drop NaN judge calls from both cal and test sets to keep things honest.
        mask = ~np.isnan(scores)
        cal_keep = cal_idx[mask[cal_idx]]
        test_keep = test_idx[mask[test_idx]]
        cal_resid = np.abs(scores[cal_keep] - labels_arr[cal_keep])
        test_resid = np.abs(scores[test_keep] - labels_arr[test_keep])
        for alpha in ALPHAS:
            q = split_conformal_quantile(cal_resid, alpha)
            if not np.isfinite(q):
                rows.append(
                    {
                        "judge": judge_id,
                        "alpha": alpha,
                        "nominal": 1 - alpha,
                        "empirical": None,
                        "q": None,
                        "n_cal": int(cal_keep.size),
                        "n_test": int(test_keep.size),
                    }
                )
                continue
            empirical = float((test_resid <= q).mean())
            rows.append(
                {
                    "judge": judge_id,
                    "alpha": alpha,
                    "nominal": 1 - alpha,
                    "empirical": empirical,
                    "q": float(q),
                    "n_cal": int(cal_keep.size),
                    "n_test": int(test_keep.size),
                }
            )
    return rows


async def main() -> None:
    t0 = time.perf_counter()
    settings = load_settings()
    openai_key = settings.api_key("OPENAI_API_KEY")
    anthropic_key = settings.api_key("ANTHROPIC_API_KEY")

    console.print("[bold]1. Loading + obfuscating HumanEval ...[/]")
    base = load_humaneval(cache_dir=settings.cache_dir, limit=None)
    probes = obfuscate_humaneval(base)[:N_ITEMS]
    console.print(f"   {len(probes)} obfuscated probes ready")

    console.print(f"[bold]2. Generating candidates ({GEN_MODEL}, temp={GEN_TEMPERATURE}) ...[/]")
    candidates = await generate_candidates(probes, settings, openai_key)
    nonempty = sum(1 for c in candidates if c.strip())
    console.print(f"   {nonempty}/{len(candidates)} non-empty candidates")

    console.print("[bold]3. Sandbox grading for ground-truth labels ...[/]")
    labels = grade(probes, candidates)
    n_pass = int(sum(labels))
    console.print(
        f"   ground truth: [green]{n_pass}/{len(labels)} pass[/], "
        f"[red]{len(labels) - n_pass} fail[/] "
        f"(target: a meaningful mix, not all-pass or all-fail)"
    )

    template = load_prompt_template(Path("prompts/rubric_code_v1.md"))
    console.print("[bold]4. Calling judges on (problem, candidate) pairs ...[/]")
    judge_scores: dict[str, list[float]] = {}
    for provider, model, env in JUDGES:
        if provider == "anthropic":
            client = AnthropicClient(api_key=anthropic_key, model=model)
        else:
            client = OpenAIClient(api_key=openai_key, model=model)
        judge = RubricJudge(client=client, template=template, variant="rubric_code_v1")
        try:
            scores = await judge_candidates(probes, candidates, judge)
        finally:
            await client.aclose()
        judge_scores[judge.judge_id] = list(scores)
        n_valid = sum(1 for s in scores if not np.isnan(s))
        console.print(f"   {judge.judge_id}: {n_valid}/{len(scores)} valid scores")

    console.print(
        "[bold]5. Fitting split conformal on calibration residuals, "
        "evaluating empirical coverage on held-out test set ...[/]"
    )
    rows = fit_and_evaluate(judge_scores, labels, seed=SEED)

    table = Table(
        title=(
            "Held-out conformal coverage on the obfuscated-HumanEval calibration probe"
        )
    )
    table.add_column("judge", style="cyan")
    table.add_column("alpha", justify="right")
    table.add_column("nominal\n(1-α)", justify="right")
    table.add_column("empirical", justify="right")
    table.add_column("|emp - nom|", justify="right")
    table.add_column("q", justify="right")
    table.add_column("n_cal", justify="right")
    table.add_column("n_test", justify="right")
    for row in rows:
        if row["empirical"] is None:
            table.add_row(
                row["judge"], f"{row['alpha']:.2f}", f"{row['nominal']:.2f}",
                "[dim]n/a[/]", "[dim]n/a[/]", "[dim]inf[/]",
                str(row["n_cal"]), str(row["n_test"]),
            )
            continue
        gap = abs(row["empirical"] - row["nominal"])
        gap_str = f"{gap*100:.1f}pp"
        if gap <= 0.05:
            gap_str = f"[green]{gap_str}[/]"
        elif gap <= 0.10:
            gap_str = f"[yellow]{gap_str}[/]"
        else:
            gap_str = f"[red]{gap_str}[/]"
        table.add_row(
            row["judge"], f"{row['alpha']:.2f}", f"{row['nominal']:.2f}",
            f"{row['empirical']:.2f}", gap_str, f"{row['q']:.3f}",
            str(row["n_cal"]), str(row["n_test"]),
        )
    console.print(table)
    console.print(
        f"\n[dim]Total wall time: {time.perf_counter() - t0:.1f}s. "
        f"Pass rate of generated candidates: {n_pass / len(labels):.0%}.[/]"
    )


if __name__ == "__main__":
    asyncio.run(main())
