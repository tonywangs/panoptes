"""Live progress check for an in-flight `panoptes eval` run.

Usage: uv run python scripts/progress.py [path/to/file.duckdb]
Default path is runs/demo_calibration.duckdb.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

DEFAULT_PATH = Path("runs/demo_calibration.duckdb")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"no duckdb at {path}")
        return
    conn = duckdb.connect(str(path), read_only=True)
    items_done = conn.execute(
        "SELECT COUNT(DISTINCT item_id) FROM eval_rows"
    ).fetchone()[0]
    rows_total = conn.execute("SELECT COUNT(*) FROM eval_rows").fetchone()[0]
    uq_total = conn.execute(
        "SELECT COUNT(*) FROM judge_uq_results"
    ).fetchone()[0]
    cost = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) FROM eval_rows"
    ).fetchone()[0]
    print(f"db:        {path}")
    print(f"items:     {items_done}")
    print(f"rows:      {rows_total}")
    print(f"uq blobs:  {uq_total}")
    print(f"cost:      ${cost:.4f}")


if __name__ == "__main__":
    main()
