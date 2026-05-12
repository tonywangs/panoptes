"""Streamlit dashboard for PANOPTES runs.

Pages:
    - **Overview** — cost tracker, run metadata, headline metrics
    - **Per-example drill-down** — one item at a time: judges, scores,
      sampling-pass UQ results
    - **Judge comparison** — paired-bootstrap rank correlation between two
      judges, permutation test for disagreement
    - **Conformal Pareto** — coverage-width sweep over α

The dashboard reads duckdb directly (no separate ETL). `st.cache_data`
caches per-query results; the 1k-row design target is comfortably met.

Run:
    uv run streamlit run src/panoptes/dashboard/app.py -- --db runs/v1.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Connection / data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def _connection(db_path: str) -> duckdb.DuckDBPyConnection:
    """One read-only duckdb connection per dashboard session."""
    return duckdb.connect(db_path, read_only=True)


@st.cache_data
def _load_runs(db_path: str) -> pd.DataFrame:
    conn = _connection(db_path)
    return cast(pd.DataFrame, conn.execute("SELECT * FROM runs ORDER BY created_at_utc DESC").df())


@st.cache_data
def _load_rows(db_path: str, run_id: str | None) -> pd.DataFrame:
    conn = _connection(db_path)
    if run_id is None:
        return cast(pd.DataFrame, conn.execute("SELECT * FROM eval_rows").df())
    return cast(
        pd.DataFrame,
        conn.execute("SELECT * FROM eval_rows WHERE run_id = ?", [run_id]).df(),
    )


@st.cache_data
def _load_uq(db_path: str, run_id: str | None) -> pd.DataFrame:
    conn = _connection(db_path)
    if run_id is None:
        return cast(pd.DataFrame, conn.execute("SELECT * FROM judge_uq_results").df())
    return cast(
        pd.DataFrame,
        conn.execute(
            "SELECT * FROM judge_uq_results WHERE run_id = ?", [run_id]
        ).df(),
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _page_overview(db_path: str, run_id: str | None) -> None:
    st.subheader("Overview")
    runs = _load_runs(db_path)
    if runs.empty:
        st.info("No runs found in this database.")
        return
    if run_id is not None:
        runs = runs[runs["run_id"] == run_id]
    st.dataframe(runs, hide_index=True, use_container_width=True)

    rows = _load_rows(db_path, run_id)
    uq = _load_uq(db_path, run_id)
    if rows.empty:
        st.info("No eval_rows for this run yet.")
        return
    point_rows = rows[rows["sample_index"] == 0]
    cols = st.columns(5)
    cols[0].metric("items", point_rows["item_id"].nunique())
    cols[1].metric("judge calls", len(rows))
    cols[2].metric("USD total", f"${rows['cost_usd'].sum():.4f}")
    cols[3].metric("UQ results", len(uq))
    cols[4].metric("benchmarks", rows["benchmark"].nunique())

    st.markdown("**Cost by judge**")
    by_judge = (
        rows.groupby("judge_id", as_index=False)["cost_usd"]
        .sum()
        .sort_values("cost_usd", ascending=False)
    )
    st.dataframe(by_judge, hide_index=True, use_container_width=True)


def _page_drilldown(db_path: str, run_id: str | None) -> None:
    st.subheader("Per-example drill-down")
    rows = _load_rows(db_path, run_id)
    uq = _load_uq(db_path, run_id)
    if rows.empty:
        st.info("No data.")
        return
    item_ids = sorted(rows["item_id"].unique())
    chosen = st.selectbox("Item", item_ids)
    if not chosen:
        return
    item_rows = rows[rows["item_id"] == chosen]
    st.markdown("**Judge responses** (point pass: `sample_index = 0`)")
    cols_to_show = [
        "judge_id", "sample_index", "score_value", "rationale",
        "input_tokens", "output_tokens", "cost_usd", "latency_ms",
    ]
    st.dataframe(item_rows[cols_to_show], hide_index=True, use_container_width=True)

    item_uq = uq[uq["item_id"] == chosen]
    if not item_uq.empty:
        st.markdown("**Per-item UQ results**")
        st.dataframe(item_uq, hide_index=True, use_container_width=True)


def _page_judge_compare(db_path: str, run_id: str | None) -> None:
    st.subheader("Judge comparison (paired bootstrap)")
    rows = _load_rows(db_path, run_id)
    point_rows = rows[rows["sample_index"] == 0]
    judges = sorted(point_rows["judge_id"].unique())
    if len(judges) < 2:
        st.info("Need ≥ 2 judges in the run to compare.")
        return
    col1, col2 = st.columns(2)
    judge_a = col1.selectbox("Judge A", judges, index=0)
    judge_b = col2.selectbox("Judge B", judges, index=1)
    if judge_a == judge_b:
        st.warning("Pick two distinct judges.")
        return
    pivot_df = point_rows.pivot_table(
        index="item_id", columns="judge_id", values="score_value", aggfunc="mean"
    )
    pair = pivot_df[[judge_a, judge_b]].dropna()
    if pair.empty:
        st.info("No items scored by both judges in this run.")
        return
    a = pair[judge_a].to_numpy(dtype=np.float64)
    b = pair[judge_b].to_numpy(dtype=np.float64)
    from panoptes.stats.compare import (  # noqa: PLC0415
        paired_bootstrap_kendall,
        paired_bootstrap_spearman,
        permutation_test_disagreement,
    )

    spearman = paired_bootstrap_spearman(a, b, rng=np.random.default_rng(0))
    kendall = paired_bootstrap_kendall(a, b, rng=np.random.default_rng(0))
    perm = permutation_test_disagreement(a, b, rng=np.random.default_rng(0))
    metrics = st.columns(3)
    metrics[0].metric(
        "Spearman ρ", f"{spearman.point:.3f}",
        delta=f"[{spearman.ci_low:.3f}, {spearman.ci_high:.3f}]",
    )
    metrics[1].metric(
        "Kendall τ", f"{kendall.point:.3f}",
        delta=f"[{kendall.ci_low:.3f}, {kendall.ci_high:.3f}]",
    )
    metrics[2].metric(
        "Disagreement p", f"{perm.p_value:.3f}", delta=f"obs={perm.observed:.3f}"
    )
    st.markdown(f"**N items compared**: {len(pair)}")
    st.scatter_chart(pair, x=judge_a, y=judge_b)


def _page_pareto(db_path: str, run_id: str | None) -> None:
    st.subheader("Conformal coverage–width Pareto (split conformal stand-in)")
    rows = _load_rows(db_path, run_id)
    point_rows = rows[rows["sample_index"] == 0]
    if point_rows.empty:
        st.info("No point-pass rows.")
        return
    # Stand-in residuals: per-item |score_j - mean across judges|.
    pivot_df = point_rows.pivot_table(
        index="item_id", columns="judge_id", values="score_value", aggfunc="mean"
    )
    if pivot_df.shape[1] < 2:
        st.info("Need ≥ 2 judges for an inter-judge residual proxy.")
        return
    means = pivot_df.mean(axis=1)
    residuals_arr: list[float] = []
    for col in pivot_df.columns:
        residuals_arr.extend(np.abs(pivot_df[col] - means).dropna().tolist())
    if not residuals_arr:
        st.info("No residuals computable.")
        return
    from panoptes.stats.pareto import coverage_width_pareto  # noqa: PLC0415

    points = coverage_width_pareto(np.asarray(residuals_arr, dtype=np.float64))
    df = pd.DataFrame(
        {
            "alpha": [p.alpha for p in points],
            "target_coverage": [p.target_coverage for p in points],
            "empirical_coverage": [p.empirical_coverage for p in points],
            "mean_width": [p.mean_width for p in points],
        }
    )
    st.line_chart(df.set_index("alpha")[["target_coverage", "empirical_coverage"]])
    st.markdown("**Mean interval width vs α**")
    st.line_chart(df.set_index("alpha")[["mean_width"]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_cli_args() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=str, help="path to PANOPTES duckdb")
    args, _ = parser.parse_known_args()
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        msg = f"DuckDB file not found: {db_path}"
        raise SystemExit(msg)
    return str(db_path)


def main() -> None:
    st.set_page_config(page_title="PANOPTES", layout="wide")
    db_path = _parse_cli_args()
    st.title("PANOPTES dashboard")
    st.caption(db_path)

    runs = _load_runs(db_path)
    if runs.empty:
        st.warning("This duckdb has no runs yet.")
        return

    run_ids: list[Any] = ["(all runs)", *runs["run_id"].tolist()]
    selected = st.sidebar.selectbox("Run", run_ids)
    run_filter: str | None = None if selected == "(all runs)" else str(selected)
    page = st.sidebar.radio(
        "Page", ["Overview", "Drill-down", "Judge comparison", "Conformal Pareto"]
    )
    if page == "Overview":
        _page_overview(db_path, run_filter)
    elif page == "Drill-down":
        _page_drilldown(db_path, run_filter)
    elif page == "Judge comparison":
        _page_judge_compare(db_path, run_filter)
    else:
        _page_pareto(db_path, run_filter)


if __name__ == "__main__":
    main()
