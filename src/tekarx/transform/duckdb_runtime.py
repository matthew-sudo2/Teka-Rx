"""Shared bounded-memory DuckDB configuration for local ETL stages."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb

_STAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def configure_duckdb(
    connection: duckdb.DuckDBPyConnection,
    *,
    data_dir: Path,
    stage: str,
    memory_limit: str,
    threads: int | None,
) -> Path:
    """Configure deterministic low-memory execution with local disk spilling."""
    if not _STAGE_NAME.fullmatch(stage):
        raise ValueError(f"invalid DuckDB stage name: {stage!r}")
    if threads is not None and threads < 1:
        raise ValueError("threads must be at least 1")

    temporary = Path(data_dir) / "interim" / ".duckdb_temp" / stage
    temporary.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET memory_limit = '{_sql_literal(memory_limit)}'")
    if threads is not None:
        connection.execute(f"SET threads = {threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = '{_sql_literal(temporary.as_posix())}'")
    effective_threads = threads if threads is not None else "DuckDB default"
    print(
        f"DuckDB {stage}: memory={memory_limit}, threads={effective_threads}, "
        f"preserve_insertion_order=false, spill={temporary}",
        flush=True,
    )
    return temporary


def _sql_literal(value: object) -> str:
    return str(value).replace("'", "''")
