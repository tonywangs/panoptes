"""Print a polished summary of the most recent live_demo_*.duckdb.

Used at the end of the recorded demo to show that the framework didn't
just call some APIs and print a cost table — it produced statistical
artifacts (per-judge disagreement, self-consistency CIs, semantic entropy
clusters when present) that the eval framework gives you on top of the
raw score.

Usage:
    uv run python scripts/show_live_run.py
    uv run python scripts/show_live_run.py path/to/specific.duckdb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _short(judge_id: str) -> str:
    parts = judge_id.split(":")
    return parts[1] if len(parts) >= 2 else judge_id


def _spread_color(spread: float) -> str:
    if spread <= 0.05:
        return "green"
    if spread <= 0.2:
        return "yellow"
    return "red"


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        candidates = sorted(Path("runs").glob("live_demo_*.duckdb"))
        if not candidates:
            console.print("[red]No runs/live_demo_*.duckdb found.[/]")
            console.print("Run the demo eval first, then re-run this script.")
            return
        path = candidates[-1]

    if not path.exists():
        console.print(f"[red]{path} not found.[/]")
        return

    console.print()
    console.print(Panel.fit(
        f"[bold]live demo analysis[/]\n"
        f"[dim]reading {path}[/]",
        border_style="green",
    ))

    conn = duckdb.connect(str(path), read_only=True)

    # 1. score table
    rows = conn.execute(
        """
        SELECT item_id, judge_id, score_value, rationale
        FROM eval_rows
        WHERE sample_index = 0
        ORDER BY item_id, judge_id
        """
    ).fetchall()
    by_item: dict[str, list[tuple[str, float, str]]] = {}
    for item_id, judge_id, score, rationale in rows:
        by_item.setdefault(item_id, []).append((judge_id, float(score), rationale or ""))
    judges = sorted({j for arr in by_item.values() for j, _, _ in arr})

    table = Table(
        title="\n[bold]point-pass scores by item, by judge[/]",
        title_justify="left",
        title_style="white",
    )
    table.add_column("item", style="cyan", no_wrap=True)
    for j in judges:
        table.add_column(_short(j), justify="right")
    table.add_column("inter-judge spread", justify="right")

    spreads: list[float] = []
    for item_id, scores in sorted(by_item.items()):
        score_map = {j: s for j, s, _ in scores}
        row: list[str] = [item_id]
        vals: list[float] = []
        for j in judges:
            s = score_map.get(j)
            row.append(f"{s:.3f}" if s is not None else "—")
            if s is not None:
                vals.append(s)
        spread = max(vals) - min(vals) if len(vals) >= 2 else 0.0
        spreads.append(spread)
        row.append(f"[{_spread_color(spread)}]{spread:.3f}[/]")
        table.add_row(*row)
    console.print(table)
    if spreads:
        mean_spread = sum(spreads) / len(spreads)
        max_spread = max(spreads)
        console.print(
            f"  [dim]mean inter-judge spread: [white]{mean_spread:.3f}[/], "
            f"max: [white]{max_spread:.3f}[/][/]\n"
        )

    # 2. UQ results
    uq_rows = conn.execute(
        """
        SELECT item_id, judge_id, method, value_json
        FROM judge_uq_results
        ORDER BY item_id, judge_id, method
        """
    ).fetchall()
    if not uq_rows:
        console.print("[dim]no UQ results in this run.[/]")
        return

    uq_table = Table(
        title="[bold]uncertainty quantification — what conventional eval throws away[/]",
        title_justify="left",
        title_style="white",
    )
    uq_table.add_column("item", style="cyan")
    uq_table.add_column("judge", style="white")
    uq_table.add_column("method", style="magenta")
    uq_table.add_column("signal", justify="left")

    for item_id, judge_id, method, value_json in uq_rows:
        if isinstance(value_json, str):
            try:
                value: dict[str, Any] = json.loads(value_json)
            except json.JSONDecodeError:
                value = {}
        else:
            value = value_json or {}

        if method == "self-consistency":
            mean = value.get("mean", 0)
            ci_low = value.get("ci_low", mean)
            ci_high = value.get("ci_high", mean)
            variance = value.get("variance", 0)
            width = ci_high - ci_low
            sig = (
                f"mean=[white]{mean:.3f}[/]  "
                f"90% CI=[white][{ci_low:.3f}, {ci_high:.3f}][/]  "
                f"width=[{'green' if width < 0.05 else 'yellow' if width < 0.15 else 'red'}]{width:.3f}[/]  "
                f"var={variance:.4f}"
            )
        elif method == "semantic-entropy":
            entropy = value.get("entropy", 0)
            n_clusters = value.get("n_clusters", 0)
            sig = (
                f"H=[white]{entropy:.3f}[/]  "
                f"clusters=[white]{n_clusters}[/]  "
                f"sizes={value.get('cluster_sizes')}"
            )
        elif method == "decomposition":
            total = value.get("total", 0)
            aleatoric = value.get("aleatoric", 0)
            epistemic = value.get("epistemic", 0)
            sig = (
                f"total=[white]{total:.4f}[/]  "
                f"aleatoric=[cyan]{aleatoric:.4f}[/]  "
                f"epistemic=[magenta]{epistemic:.4f}[/]"
            )
        else:
            preview = json.dumps(value)[:60]
            sig = f"[dim]{preview}…[/]"

        uq_table.add_row(item_id, _short(judge_id), method, sig)
    console.print(uq_table)

    # 3. punchline
    interesting_items: list[tuple[str, float]] = []
    for item_id, scores in by_item.items():
        if len(scores) >= 2:
            vals = [s for _, s, _ in scores]
            interesting_items.append((item_id, max(vals) - min(vals)))
    interesting_items.sort(key=lambda x: x[1], reverse=True)

    if interesting_items:
        most_uncertain = interesting_items[0]
        most_certain = interesting_items[-1]
        console.print(
            Panel(
                (
                    f"[bold]the framework's verdict[/]\n\n"
                    f"  most disagreement: [cyan]{most_uncertain[0]}[/] "
                    f"  spread = [red]{most_uncertain[1]:.3f}[/]  "
                    f"[dim]→ a bandit would escalate to a stronger judge here[/]\n"
                    f"  least disagreement: [cyan]{most_certain[0]}[/] "
                    f"  spread = [green]{most_certain[1]:.3f}[/]  "
                    f"[dim]→ a bandit would stop calling more judges here[/]"
                ),
                border_style="green",
                title="[bold green]not just scores · actionable uncertainty[/]",
                title_align="left",
                padding=(1, 2),
            )
        )

    conn.close()


if __name__ == "__main__":
    main()
