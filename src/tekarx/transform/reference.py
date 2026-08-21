"""Shared records and verification for reference-data Parquet builds."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ReferenceBuildRecord:
    """Provenance for one raw-to-interim reference table."""

    dataset: str
    table: str
    source_path: str
    source_sha256: str
    output_path: str
    rows: int
    columns: int
    output_size_bytes: int
    compression: str = "snappy"
    cached: bool = False


def verify_parquet(path: Path, *, expected_rows: int, expected_schema: pa.Schema) -> None:
    """Verify row count, schema metadata, and compression before promotion."""
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise RuntimeError(
            f"row-count mismatch for {path}: {parquet.metadata.num_rows} != {expected_rows}"
        )
    if not parquet.schema_arrow.equals(expected_schema, check_metadata=True):
        raise RuntimeError(f"schema mismatch for {path}")
    for row_group in range(parquet.metadata.num_row_groups):
        for column in range(parquet.metadata.num_columns):
            compression = parquet.metadata.row_group(row_group).column(column).compression
            if compression != "SNAPPY":
                raise RuntimeError(f"non-Snappy column found in {path}")


def cached_record(
    *, destination: Path, source: Path, source_sha256: str, dataset: str, table: str
) -> ReferenceBuildRecord | None:
    """Return a verified cached record or reject stale output."""
    if not destination.is_file():
        return None
    parquet = pq.ParquetFile(destination)
    metadata = parquet.schema_arrow.metadata or {}
    expected = {
        b"tekarx.source_sha256": source_sha256.encode(),
        b"tekarx.dataset": dataset.encode(),
        b"tekarx.table": table.encode(),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"existing Parquet has incompatible provenance: {destination}")
    return ReferenceBuildRecord(
        dataset=dataset,
        table=table,
        source_path=str(source),
        source_sha256=source_sha256,
        output_path=str(destination),
        rows=parquet.metadata.num_rows,
        columns=parquet.metadata.num_columns,
        output_size_bytes=destination.stat().st_size,
        cached=True,
    )


def write_manifest(path: Path, *, dataset: str, records: list[ReferenceBuildRecord]) -> None:
    """Atomically upsert build records by table."""
    if path.is_file():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid interim manifest: {path}") from exc
    else:
        manifest = {"dataset": dataset, "artifacts": []}
    if manifest.get("dataset") != dataset or not isinstance(manifest.get("artifacts"), list):
        raise RuntimeError(f"invalid interim manifest structure: {path}")
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    tables = {record.table for record in records}
    artifacts[:] = [
        item
        for item in artifacts
        if not isinstance(item, dict) or item.get("table") not in tables
    ]
    artifacts.extend(asdict(record) for record in records)
    artifacts.sort(key=lambda item: str(item.get("table")) if isinstance(item, dict) else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
