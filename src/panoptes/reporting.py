"""Offline HTML report.

Reads a PANOPTES duckdb file and emits a self-contained HTML page with run
metadata, cost-by-judge, UQ-result counts, and inter-judge agreement.
Reliability / coverage diagnostics that need ground-truth labels are
populated when the calibration probe is run.

The output has no JS dependencies. For interactive exploration use the
Streamlit dashboard.
"""

from __future__ import annotations

import html
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from panoptes.stats.compare import paired_bootstrap_spearman


def _esc(value: object) -> str:
    return html.escape(str(value))


def _section(title: str, body: str) -> str:
    return f"<section><h2>{_esc(title)}</h2>{body}</section>"


def _table_from_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>(no rows)</em></p>"
    return df.to_html(index=False, border=0, classes="panoptes")


def _runs_section(conn: duckdb.DuckDBPyConnection) -> str:
    df = conn.execute("SELECT * FROM runs ORDER BY created_at_utc DESC").df()
    return _section("Runs", _table_from_df(df))


def _cost_section(conn: duckdb.DuckDBPyConnection) -> str:
    df = conn.execute(
        """
        SELECT judge_id,
               COUNT(*) AS n_calls,
               SUM(cost_usd) AS usd_total,
               SUM(input_tokens) AS input_tokens,
               SUM(output_tokens) AS output_tokens,
               SUM(cache_read_tokens) AS cache_read_tokens
        FROM eval_rows
        GROUP BY judge_id
        ORDER BY usd_total DESC
        """
    ).df()
    return _section("Cost by judge", _table_from_df(df))


def _uq_section(conn: duckdb.DuckDBPyConnection) -> str:
    df = conn.execute(
        """
        SELECT method, COUNT(*) AS n
        FROM judge_uq_results
        GROUP BY method
        ORDER BY n DESC
        """
    ).df()
    return _section("UQ results by method", _table_from_df(df))


def _agreement_section(conn: duckdb.DuckDBPyConnection) -> str:
    pivot = conn.execute(
        """
        SELECT item_id, judge_id, AVG(score_value) AS score
        FROM eval_rows
        WHERE sample_index = 0
        GROUP BY item_id, judge_id
        """
    ).df()
    if pivot.empty:
        return _section("Inter-judge agreement", "<p><em>(no point-pass rows)</em></p>")
    wide = pivot.pivot(index="item_id", columns="judge_id", values="score")
    judges = list(wide.columns)
    if len(judges) < 2:
        return _section("Inter-judge agreement", "<p><em>(need ≥ 2 judges)</em></p>")
    rows: list[tuple[str, str, int, float, float, float]] = []
    for i, ja in enumerate(judges):
        for jb in judges[i + 1 :]:
            pair = wide[[ja, jb]].dropna()
            if len(pair) < 5:
                continue
            a = np.asarray(pair[ja].values, dtype=np.float64)
            b = np.asarray(pair[jb].values, dtype=np.float64)
            corr = paired_bootstrap_spearman(a, b, rng=np.random.default_rng(0))
            rows.append((str(ja), str(jb), len(pair), corr.point, corr.ci_low, corr.ci_high))
    if not rows:
        return _section("Inter-judge agreement", "<p><em>(not enough shared items)</em></p>")
    df = pd.DataFrame(
        rows, columns=["judge_a", "judge_b", "n", "spearman", "ci_low", "ci_high"]
    )
    return _section("Inter-judge agreement (Spearman ρ ± 90% CI)", _table_from_df(df))


_BASE_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 2rem; max-width: 1200px; margin: 0 auto; }
section { margin-bottom: 2rem; }
table.panoptes { border-collapse: collapse; width: 100%; }
table.panoptes th, table.panoptes td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; text-align: left; }
table.panoptes th { background: #fafafa; }
"""


def render_report(db_path: Path) -> str:
    """Render the full report as a single HTML string."""
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        sections = [
            _runs_section(conn),
            _cost_section(conn),
            _uq_section(conn),
            _agreement_section(conn),
        ]
    finally:
        conn.close()
    return (
        "<!doctype html><html><head>"
        '<meta charset="utf-8"><title>PANOPTES report</title>'
        f"<style>{_BASE_CSS}</style></head><body>"
        f"<h1>PANOPTES report</h1><p>{_esc(db_path)}</p>"
        + "".join(sections)
        + "</body></html>"
    )


def write_report(db_path: Path, output: Path) -> None:
    output.write_text(render_report(db_path), encoding="utf-8")


__all__ = ["render_report", "write_report"]
