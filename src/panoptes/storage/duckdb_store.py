"""DuckDB-backed result store.

Schema (v1, M1):

    runs(
        run_id           TEXT PRIMARY KEY,
        created_at_utc   TIMESTAMP,
        config_json      JSON,
        panoptes_version TEXT
    )

    eval_rows(
        run_id                 TEXT,
        item_id                TEXT,
        benchmark              TEXT,
        task_family            TEXT,
        judge_id               TEXT,
        prompt_version_hash    TEXT,
        model_under_test       TEXT,
        model_response         TEXT,
        score_value            DOUBLE,
        score_scale            TEXT,
        likert                 INTEGER,
        rationale              TEXT,
        flags_json             JSON,
        input_tokens           INTEGER,
        output_tokens          INTEGER,
        cache_read_tokens      INTEGER,
        cache_creation_tokens  INTEGER,
        cost_usd               DOUBLE,
        latency_ms             DOUBLE,
        raw_text               TEXT,
        sample_index           INTEGER,
        temperature            DOUBLE,
        timestamp_utc          TIMESTAMP,
        PRIMARY KEY (run_id, item_id, judge_id, sample_index)
    )

The (task_family, judge_id, prompt_version_hash) tuple is the logical
partition key the spec calls for; duckdb stores it physically in one file but
we add an INDEX over it so partition-style queries remain fast.

We deliberately *do not* serialize the full `EvalRecord` as a nested JSON
column. Flat columns let us push aggregations (mean cost per judge, etc.)
down to duckdb where they belong. The two truly variable-shape fields
(`flags_json`, `config_json`) are stored as JSON.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from panoptes._version import __version__
from panoptes.errors import StorageError
from panoptes.schemas import EvalRecord, JudgeResponse

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


_SCHEMA_VERSION = 1


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    created_at_utc   TIMESTAMP NOT NULL,
    config_json      JSON,
    panoptes_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_rows (
    run_id                 TEXT NOT NULL,
    item_id                TEXT NOT NULL,
    benchmark              TEXT NOT NULL,
    task_family            TEXT NOT NULL,
    judge_id               TEXT NOT NULL,
    prompt_version_hash    TEXT NOT NULL,
    model_under_test       TEXT NOT NULL,
    model_response         TEXT NOT NULL,
    score_value            DOUBLE NOT NULL,
    score_scale            TEXT NOT NULL,
    likert                 INTEGER,
    rationale              TEXT NOT NULL,
    flags_json             JSON,
    input_tokens           INTEGER NOT NULL,
    output_tokens          INTEGER NOT NULL,
    cache_read_tokens      INTEGER NOT NULL,
    cache_creation_tokens  INTEGER NOT NULL,
    cost_usd               DOUBLE NOT NULL,
    latency_ms             DOUBLE NOT NULL,
    raw_text               TEXT NOT NULL,
    sample_index           INTEGER NOT NULL,
    temperature            DOUBLE NOT NULL,
    timestamp_utc          TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, item_id, judge_id, sample_index)
);

CREATE INDEX IF NOT EXISTS idx_eval_rows_partition
    ON eval_rows (task_family, judge_id, prompt_version_hash);
"""


@dataclass(frozen=True, slots=True)
class JudgeRow:
    """Flat row materialized from an `EvalRecord` + `JudgeResponse` pair.

    The store layer works in terms of these flat rows; the pipeline builds
    them from richer Pydantic models. Keeping the storage interface flat
    makes pyright-strict adapters trivial and avoids a json-everywhere
    schema mess.
    """

    run_id: str
    item_id: str
    benchmark: str
    task_family: str
    judge_id: str
    prompt_version_hash: str
    model_under_test: str
    model_response: str
    score_value: float
    score_scale: str
    likert: int | None
    rationale: str
    flags_json: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float
    latency_ms: float
    raw_text: str
    sample_index: int
    temperature: float
    timestamp_utc: datetime


def row_from_response(record: EvalRecord, judge: JudgeResponse) -> JudgeRow:
    """Materialize one `JudgeRow` from a parent `EvalRecord` and one judge call."""
    return JudgeRow(
        run_id=record.run_id,
        item_id=record.item.item_id,
        benchmark=record.item.benchmark,
        task_family=record.item.task_family.value,
        judge_id=judge.judge_id,
        prompt_version_hash=judge.prompt_hash,
        model_under_test=record.model_under_test,
        model_response=record.model_response,
        score_value=judge.score.value,
        score_scale=judge.score.scale.value,
        likert=judge.score.likert,
        rationale=judge.score.rationale,
        flags_json=json.dumps(judge.score.flags),
        input_tokens=judge.usage.input_tokens,
        output_tokens=judge.usage.output_tokens,
        cache_read_tokens=judge.usage.cache_read_tokens,
        cache_creation_tokens=judge.usage.cache_creation_tokens,
        cost_usd=judge.cost_usd,
        latency_ms=judge.latency_ms,
        raw_text=judge.raw_text,
        sample_index=judge.sample_index,
        temperature=judge.sampled_at_temperature,
        timestamp_utc=judge.timestamp_utc,
    )


class DuckDBStore:
    """Persistent result store. Holds one open duckdb connection.

    Instantiate via `DuckDBStore.open(path)`; use as a context manager or
    call `close()` explicitly when finished.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, path: Path) -> None:
        self._conn = conn
        self._path = path

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def open(cls, path: Path) -> DuckDBStore:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(path))
        store = cls(conn, path)
        store._init_schema()
        return store

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DuckDBStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ schema

    def _init_schema(self) -> None:
        self._conn.execute(_CREATE_SQL)
        # Record / verify the schema version. M1 is version 1; future
        # migrations bump this and apply DDL deltas.
        result = self._conn.execute("SELECT version FROM schema_meta").fetchone()
        if result is None:
            self._conn.execute("INSERT INTO schema_meta (version) VALUES (?)", [_SCHEMA_VERSION])
            return
        existing = int(result[0])
        if existing != _SCHEMA_VERSION:
            raise StorageError(
                f"duckdb at {self._path} has schema version {existing}, "
                f"this build expects {_SCHEMA_VERSION}. Migration not yet implemented."
            )

    # ------------------------------------------------------------------ writes

    def record_run(self, *, run_id: str, config: dict[str, Any]) -> None:
        """Insert (or replace) one row in the `runs` table."""
        self._conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?)",
            [run_id, datetime.now(UTC), json.dumps(config), __version__],
        )

    def write_rows(self, rows: Iterable[JudgeRow]) -> int:
        """Idempotent bulk-insert of `JudgeRow`s; returns number written."""
        n = 0
        with self._transaction():
            for row in rows:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO eval_rows VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        row.run_id,
                        row.item_id,
                        row.benchmark,
                        row.task_family,
                        row.judge_id,
                        row.prompt_version_hash,
                        row.model_under_test,
                        row.model_response,
                        row.score_value,
                        row.score_scale,
                        row.likert,
                        row.rationale,
                        row.flags_json,
                        row.input_tokens,
                        row.output_tokens,
                        row.cache_read_tokens,
                        row.cache_creation_tokens,
                        row.cost_usd,
                        row.latency_ms,
                        row.raw_text,
                        row.sample_index,
                        row.temperature,
                        row.timestamp_utc,
                    ],
                )
                n += 1
        return n

    # ------------------------------------------------------------------ reads

    def count_rows(self) -> int:
        result = self._conn.execute("SELECT COUNT(*) FROM eval_rows").fetchone()
        return 0 if result is None else int(result[0])

    def has_cached(
        self,
        *,
        benchmark: str,
        judge_id: str,
        item_id: str,
        prompt_version_hash: str,
    ) -> bool:
        """Return True iff a row already exists for this (judge, item, prompt) tuple.

        Cross-run cache: any prior successful evaluation of this combination
        counts as a hit, regardless of which `run_id` produced it.
        """
        result = self._conn.execute(
            """
            SELECT 1 FROM eval_rows
            WHERE benchmark = ? AND judge_id = ? AND item_id = ?
              AND prompt_version_hash = ?
              AND sample_index = 0
            LIMIT 1
            """,
            [benchmark, judge_id, item_id, prompt_version_hash],
        ).fetchone()
        return result is not None

    # ------------------------------------------------------------------ internal

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        self._conn.execute("BEGIN TRANSACTION")
        try:
            yield
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")
