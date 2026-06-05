"""Export everything the Next.js dashboard needs as static JSON.

Reads every `runs/*.duckdb` plus `runs/calibration_results.json` and emits
self-contained JSON files to `web/public/data/`. The Next.js build then
ships those files as static assets — no runtime database, no API routes,
nothing fancy at deploy time.

Pre-computes the expensive stats (pairwise judge correlations with bootstrap
CIs, coverage-width Pareto sweeps) so the dashboard is purely a renderer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from panoptes.benchmarks.calibration_probe import obfuscate_humaneval
from panoptes.benchmarks.humaneval import load_humaneval
from panoptes.config import load_settings
from panoptes.stats.compare import (
    paired_bootstrap_kendall,
    paired_bootstrap_spearman,
    permutation_test_disagreement,
)
from panoptes.stats.pareto import coverage_width_pareto

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "runs"
OUT_DIR = REPO / "web" / "public" / "data"


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy / dataclass shapes."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        # Standard JSON doesn't support NaN/Inf; emit None so JSON.parse works
        # in the browser.
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.ndarray):
        return [None if (np.isnan(v) or np.isinf(v)) else float(v) if isinstance(v, np.floating) else v for v in obj.tolist()]
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"unserializable: {type(obj).__name__}")


def _scrub_nans(obj: Any) -> Any:
    """Recursively replace NaN / Inf floats with None so JSON.parse accepts the output."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, dict):
        return {k: _scrub_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nans(v) for v in obj]
    return obj


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scrubbed = _scrub_nans(obj)
    path.write_text(json.dumps(scrubbed, default=_json_default, indent=2, allow_nan=False))
    print(f"  wrote {path.relative_to(REPO)}  ({path.stat().st_size:,} bytes)")


def _load_runs(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT run_id, created_at_utc, config_json, panoptes_version FROM runs"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for run_id, created, config_json, version in rows:
        config = json.loads(config_json) if config_json else {}
        out.append(
            {
                "run_id": run_id,
                "created_at_utc": created.isoformat() if created else None,
                "panoptes_version": version,
                "config": config,
            }
        )
    return out


def _load_eval_rows(
    conn: duckdb.DuckDBPyConnection, run_id: str
) -> list[dict[str, Any]]:
    cols = [
        "item_id", "benchmark", "task_family", "judge_id",
        "prompt_version_hash", "model_under_test", "model_response",
        "score_value", "score_scale", "likert", "rationale", "flags_json",
        "input_tokens", "output_tokens", "cache_read_tokens",
        "cache_creation_tokens", "cost_usd", "latency_ms",
        "sample_index", "temperature", "timestamp_utc",
    ]
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM eval_rows WHERE run_id = ? ORDER BY item_id, judge_id, sample_index",
        [run_id],
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        record = dict(zip(cols, r, strict=True))
        # JSON columns: turn the string into the actual object
        if isinstance(record.get("flags_json"), str):
            try:
                record["flags"] = json.loads(record["flags_json"])
            except json.JSONDecodeError:
                record["flags"] = []
        else:
            record["flags"] = []
        record.pop("flags_json", None)
        ts = record.get("timestamp_utc")
        record["timestamp_utc"] = ts.isoformat() if ts is not None and hasattr(ts, "isoformat") else ts
        out.append(record)
    return out


def _load_uq_results(
    conn: duckdb.DuckDBPyConnection, run_id: str
) -> list[dict[str, Any]]:
    # Older duckdb files (schema v1) don't have judge_uq_results yet; treat
    # absence as empty rather than crashing the export.
    has_table = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'judge_uq_results'"
    ).fetchone()
    if not has_table or int(has_table[0]) == 0:
        return []
    rows = conn.execute(
        "SELECT item_id, judge_id, method, value_json FROM judge_uq_results WHERE run_id = ?",
        [run_id],
    ).fetchall()
    out: list[dict[str, Any]] = []
    for item_id, judge_id, method, value_json in rows:
        try:
            value = json.loads(value_json) if isinstance(value_json, str) else value_json
        except json.JSONDecodeError:
            value = None
        out.append(
            {"item_id": item_id, "judge_id": judge_id, "method": method, "value": value}
        )
    return out


def _summarize_run(
    run: dict[str, Any], rows: list[dict[str, Any]], uq: list[dict[str, Any]]
) -> dict[str, Any]:
    point_rows = [r for r in rows if r.get("sample_index", 0) == 0]
    n_items = len({r["item_id"] for r in point_rows})
    judges = sorted({r["judge_id"] for r in rows})
    cost_by_judge: dict[str, float] = defaultdict(float)
    for r in rows:
        cost_by_judge[r["judge_id"]] += float(r["cost_usd"] or 0)
    tokens: dict[str, int] = defaultdict(int)
    for r in rows:
        tokens["input"] += int(r["input_tokens"] or 0)
        tokens["output"] += int(r["output_tokens"] or 0)
        tokens["cache_read"] += int(r["cache_read_tokens"] or 0)
        tokens["cache_creation"] += int(r["cache_creation_tokens"] or 0)
    return {
        "run_id": run["run_id"],
        "created_at_utc": run.get("created_at_utc"),
        "panoptes_version": run.get("panoptes_version"),
        "config": run.get("config", {}),
        "strategy": run.get("config", {}).get("strategy", "all"),
        "n_items": n_items,
        "n_calls": len(rows),
        "n_judges": len(judges),
        "judges": judges,
        "cost_usd": sum(cost_by_judge.values()),
        "cost_by_judge": dict(cost_by_judge),
        "tokens": dict(tokens),
        "n_uq_results": len(uq),
    }


def _judge_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pre-compute pairwise judge stats for the scatter / correlation matrix."""
    point_rows = [r for r in rows if r.get("sample_index", 0) == 0]
    # item_id -> judge_id -> score (mean over duplicates)
    pivot: dict[str, dict[str, float]] = defaultdict(dict)
    for r in point_rows:
        pivot[r["item_id"]][r["judge_id"]] = float(r["score_value"])
    judges = sorted({jid for inner in pivot.values() for jid in inner.keys()})
    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            common_items = [iid for iid, inner in pivot.items() if a in inner and b in inner]
            if len(common_items) < 3:
                continue
            arr_a = np.asarray([pivot[iid][a] for iid in common_items], dtype=np.float64)
            arr_b = np.asarray([pivot[iid][b] for iid in common_items], dtype=np.float64)
            spear = paired_bootstrap_spearman(arr_a, arr_b, rng=rng)
            kend = paired_bootstrap_kendall(arr_a, arr_b, rng=rng)
            perm = permutation_test_disagreement(arr_a, arr_b, rng=rng)
            out.append(
                {
                    "judge_a": a,
                    "judge_b": b,
                    "n_items": len(common_items),
                    "spearman": {"point": spear.point, "ci_low": spear.ci_low, "ci_high": spear.ci_high},
                    "kendall": {"point": kend.point, "ci_low": kend.ci_low, "ci_high": kend.ci_high},
                    "permutation": {"observed": perm.observed, "p_value": perm.p_value},
                    "pairs": [
                        {"item_id": iid, "a": float(pivot[iid][a]), "b": float(pivot[iid][b])}
                        for iid in common_items
                    ],
                }
            )
    return out


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inter-judge spread → coverage-width Pareto sweep over alpha."""
    point_rows = [r for r in rows if r.get("sample_index", 0) == 0]
    pivot: dict[str, dict[str, float]] = defaultdict(dict)
    for r in point_rows:
        pivot[r["item_id"]][r["judge_id"]] = float(r["score_value"])
    residuals: list[float] = []
    for inner in pivot.values():
        scores = np.asarray(list(inner.values()), dtype=np.float64)
        if scores.size < 2:
            continue
        mean = float(scores.mean())
        residuals.extend(float(abs(s - mean)) for s in scores)
    if len(residuals) < 10:
        return []
    points = coverage_width_pareto(np.asarray(residuals, dtype=np.float64))
    return [
        {
            "alpha": p.alpha,
            "target_coverage": p.target_coverage,
            "mean_width": p.mean_width,
            "empirical_coverage": p.empirical_coverage,
        }
        for p in points
    ]


def _benchmark_prompts(rows_by_run: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, str]]:
    """Re-load benchmark prompts so the drill-down page can render them.

    Returns {item_id: {prompt, canonical_solution, entry_point}}.
    Only HumanEval + calibration probe items are loaded for now.
    """
    settings = load_settings()
    needed_item_ids: set[str] = set()
    for rows in rows_by_run.values():
        for r in rows:
            needed_item_ids.add(r["item_id"])
    if not needed_item_ids:
        return {}
    base = load_humaneval(cache_dir=settings.cache_dir, limit=None)
    out: dict[str, dict[str, str]] = {}
    for item in base:
        out[item.item_id] = {
            "prompt": item.prompt,
            "canonical_solution": str(item.metadata.get("canonical_solution", "")),
            "entry_point": str(item.metadata.get("entry_point", "")),
            "source": "humaneval",
        }
    # Also expose obfuscated probe items under their calib:: ids.
    for probe in obfuscate_humaneval(base):
        out[probe.item.item_id] = {
            "prompt": probe.item.prompt,
            "canonical_solution": str(probe.item.metadata.get("canonical_solution", "")),
            "entry_point": probe.rewritten_entry_point,
            "source": "humaneval-calibprobe",
            "original_entry_point": probe.original_entry_point,
        }
    return {iid: out[iid] for iid in needed_item_ids if iid in out}


def _discover_duckdbs() -> list[Path]:
    return sorted(p for p in RUNS_DIR.glob("*.duckdb") if not p.name.endswith(".wal"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing JSON to {OUT_DIR.relative_to(REPO)}/")

    duckdbs = _discover_duckdbs()
    if not duckdbs:
        print("no duckdb files in runs/, nothing to export")
        return

    all_summaries: list[dict[str, Any]] = []
    rows_by_run: dict[str, list[dict[str, Any]]] = {}

    for db_path in duckdbs:
        print(f"\nreading {db_path.relative_to(REPO)}")
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            runs = _load_runs(conn)
            for run in runs:
                run_id = run["run_id"]
                rows = _load_eval_rows(conn, run_id)
                uq = _load_uq_results(conn, run_id)
                rows_by_run[run_id] = rows
                summary = _summarize_run(run, rows, uq)
                summary["source_file"] = db_path.name
                all_summaries.append(summary)

                _write(OUT_DIR / f"rows-{run_id}.json", rows)
                _write(OUT_DIR / f"uq-{run_id}.json", uq)
                _write(OUT_DIR / f"judge-pairs-{run_id}.json", _judge_pairs(rows))
                _write(OUT_DIR / f"pareto-{run_id}.json", _pareto(rows))
        finally:
            conn.close()

    _write(OUT_DIR / "runs.json", sorted(all_summaries, key=lambda s: s.get("created_at_utc") or ""))

    print("\nloading benchmark prompts for drill-down ...")
    prompts = _benchmark_prompts(rows_by_run)
    _write(OUT_DIR / "items.json", prompts)

    calib_path = RUNS_DIR / "calibration_results.json"
    if calib_path.exists():
        _write(OUT_DIR / "calibration.json", json.loads(calib_path.read_text()))
    else:
        print(f"  (no {calib_path.name}, skipping calibration export)")

    summary_top: dict[str, Any] = {
        "n_runs": len(all_summaries),
        "n_items_total": sum(s["n_items"] for s in all_summaries),
        "n_calls_total": sum(s["n_calls"] for s in all_summaries),
        "cost_total_usd": sum(s["cost_usd"] for s in all_summaries),
        "judges_seen": sorted({j for s in all_summaries for j in s["judges"]}),
    }
    if (OUT_DIR / "calibration.json").exists():
        calib = json.loads((OUT_DIR / "calibration.json").read_text())
        headline = calib.get("headline")
        if headline:
            summary_top["calibration_headline"] = headline
    _write(OUT_DIR / "summary.json", summary_top)

    print("\ndone")


if __name__ == "__main__":
    main()
