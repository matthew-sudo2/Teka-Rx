"""Atomic Parquet output."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(
    table: pa.Table,
    destination: Path,
    *,
    compression: str = "snappy",
) -> None:
    """Write a table atomically with a stable compression default."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        pq.write_table(table, temporary, compression=compression)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
