"""Stream modern FAERS ASCII tables into provenance-preserving Parquet."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from tekarx.extract.common import sha256_file

CORE_FAERS_TABLES = ("demo", "drug", "indi", "reac", "outc", "delete")
CSV_BLOCK_SIZE = 16 * 1024 * 1024


class FaersBuildError(RuntimeError):
    """Raised when a FAERS source cannot be converted without data loss."""


@dataclass(frozen=True)
class ParquetBuildRecord:
    quarter: str
    table: str
    source_path: str
    source_sha256: str
    output_path: str
    rows: int
    columns: int
    output_size_bytes: int
    compression: str
    cached: bool = False


def build_faers(
    *, data_dir: Path, quarters: list[str], tables: tuple[str, ...] = CORE_FAERS_TABLES
) -> list[ParquetBuildRecord]:
    """Convert selected extracted quarters and update the interim build manifest."""
    invalid = set(tables) - set(CORE_FAERS_TABLES)
    if invalid:
        raise ValueError(f"unsupported FAERS tables: {sorted(invalid)}")
    records: list[ParquetBuildRecord] = []
    for quarter in quarters:
        for table in tables:
            records.append(_build_table(data_dir=data_dir, quarter=quarter, table=table))
    _write_build_manifest(data_dir / "interim" / "faers" / "manifest.json", records)
    return records


def _build_table(*, data_dir: Path, quarter: str, table: str) -> ParquetBuildRecord:
    source = _find_source_file(data_dir=data_dir, quarter=quarter, table=table)
    checksum = sha256_file(source)
    destination = data_dir / "interim" / "faers" / table / f"{quarter}.parquet"
    cached = _cached_record(
        destination=destination,
        source=source,
        source_sha256=checksum,
        quarter=quarter,
        table=table,
    )
    if cached is not None:
        return cached

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows, columns = _convert_source(
            source=source,
            destination=destination,
            source_sha256=checksum,
            quarter=quarter,
            table=table,
            encoding="utf8",
        )
    except pa.ArrowInvalid:
        print(f"WARNING: {source.name} is not valid UTF-8; retrying as Latin-1.")
        rows, columns = _convert_source(
            source=source,
            destination=destination,
            source_sha256=checksum,
            quarter=quarter,
            table=table,
            encoding="latin1",
        )
    return ParquetBuildRecord(
        quarter=quarter,
        table=table,
        source_path=str(source),
        source_sha256=checksum,
        output_path=str(destination),
        rows=rows,
        columns=columns,
        output_size_bytes=destination.stat().st_size,
        compression="snappy",
    )


def _find_source_file(*, data_dir: Path, quarter: str, table: str) -> Path:
    extracted = data_dir / "raw" / "faers" / quarter / "extracted"
    if not (extracted / ".complete").is_file():
        raise FaersBuildError(
            f"FAERS {quarter} is not completely extracted; run "
            f"`tekarx extract-faers --quarter {quarter}` first"
        )
    prefix = table.upper()
    matches = sorted(
        path
        for path in extracted.rglob("*")
        if path.is_file() and path.suffix.lower() == ".txt" and path.stem.upper().startswith(prefix)
    )
    if len(matches) != 1:
        raise FaersBuildError(f"expected one {prefix} TXT file for {quarter}, found {len(matches)}")
    return matches[0]


def _convert_source(
    *,
    source: Path,
    destination: Path,
    source_sha256: str,
    quarter: str,
    table: str,
    encoding: str,
) -> tuple[int, int]:
    columns, skip_rows = _source_layout(source, table=table)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    reader = pacsv.open_csv(
        source,
        read_options=pacsv.ReadOptions(
            block_size=CSV_BLOCK_SIZE,
            skip_rows=skip_rows,
            column_names=columns,
            encoding=encoding,
        ),
        parse_options=pacsv.ParseOptions(delimiter="$", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={column: pa.string() for column in columns},
            strings_can_be_null=True,
            null_values=[""],
        ),
    )
    metadata = {
        b"tekarx.source_path": str(source).encode(),
        b"tekarx.source_sha256": source_sha256.encode(),
        b"tekarx.quarter": quarter.encode(),
        b"tekarx.table": table.encode(),
        b"tekarx.encoding": encoding.encode(),
        b"tekarx.built_at_utc": datetime.now(UTC).isoformat().encode(),
    }
    output_schema = reader.schema.append(pa.field("quarter", pa.string(), nullable=False))
    output_schema = output_schema.with_metadata(metadata)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        writer = pq.ParquetWriter(temporary, output_schema, compression="snappy")
        progress = tqdm(desc=f"Building {quarter} {table}", unit="rows", unit_scale=True)
        try:
            for batch in reader:
                batch = batch.append_column(
                    pa.field("quarter", pa.string(), nullable=False),
                    pa.array([quarter] * batch.num_rows),
                )
                writer.write_batch(batch)
                rows += batch.num_rows
                progress.update(batch.num_rows)
        finally:
            progress.close()
        writer.close()
        writer = None
        _verify_parquet(temporary, expected_rows=rows, expected_schema=output_schema)
        os.replace(temporary, destination)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return rows, len(output_schema)


def _source_layout(source: Path, *, table: str) -> tuple[list[str], int]:
    with source.open("rb") as stream:
        header = stream.readline().decode("utf-8-sig").rstrip("\r\n")
    if table == "delete":
        if header.strip():
            raise FaersBuildError(f"unexpected content before deletion IDs in {source}")
        return ["caseid"], 1
    columns = [column.strip().lower() for column in header.split("$")]
    if not columns or any(not column for column in columns) or len(columns) != len(set(columns)):
        raise FaersBuildError(f"invalid or duplicate header columns in {source}")
    return columns, 1


def _verify_parquet(path: Path, *, expected_rows: int, expected_schema: pa.Schema) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise FaersBuildError(
            f"row-count mismatch for {path}: {parquet.metadata.num_rows} != {expected_rows}"
        )
    if not parquet.schema_arrow.equals(expected_schema, check_metadata=True):
        raise FaersBuildError(f"schema mismatch for {path}")
    for row_group in range(parquet.metadata.num_row_groups):
        for column in range(parquet.metadata.num_columns):
            if parquet.metadata.row_group(row_group).column(column).compression != "SNAPPY":
                raise FaersBuildError(f"non-Snappy column found in {path}")


def _cached_record(
    *,
    destination: Path,
    source: Path,
    source_sha256: str,
    quarter: str,
    table: str,
) -> ParquetBuildRecord | None:
    if not destination.is_file():
        return None
    parquet = pq.ParquetFile(destination)
    metadata = parquet.schema_arrow.metadata or {}
    if metadata.get(b"tekarx.source_sha256") != source_sha256.encode():
        raise FaersBuildError(
            f"existing Parquet was built from different source bytes: {destination}"
        )
    if metadata.get(b"tekarx.quarter") != quarter.encode():
        raise FaersBuildError(f"existing Parquet has the wrong quarter: {destination}")
    if metadata.get(b"tekarx.table") != table.encode():
        raise FaersBuildError(f"existing Parquet has the wrong table: {destination}")
    return ParquetBuildRecord(
        quarter=quarter,
        table=table,
        source_path=str(source),
        source_sha256=source_sha256,
        output_path=str(destination),
        rows=parquet.metadata.num_rows,
        columns=parquet.metadata.num_columns,
        output_size_bytes=destination.stat().st_size,
        compression="snappy",
        cached=True,
    )


def _write_build_manifest(path: Path, records: list[ParquetBuildRecord]) -> None:
    existing: dict[str, object]
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FaersBuildError(f"invalid interim FAERS manifest: {path}") from exc
    else:
        existing = {"dataset": "FAERS interim Parquet", "artifacts": []}
    artifacts = existing.get("artifacts")
    if not isinstance(artifacts, list):
        raise FaersBuildError(f"invalid interim FAERS manifest structure: {path}")
    keys = {(record.quarter, record.table) for record in records}
    artifacts[:] = [
        item
        for item in artifacts
        if not isinstance(item, dict) or (item.get("quarter"), item.get("table")) not in keys
    ]
    artifacts.extend(asdict(record) for record in records)
    artifacts.sort(
        key=lambda item: (
            (str(item.get("quarter")), str(item.get("table")))
            if isinstance(item, dict)
            else ("", "")
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
