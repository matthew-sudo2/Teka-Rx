"""Stream mapping-relevant tables from a DrugCentral PostgreSQL dump."""

from __future__ import annotations

import gzip
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from tekarx.extract.common import sha256_file
from tekarx.transform.reference import (
    ReferenceBuildRecord,
    cached_record,
    verify_parquet,
    write_manifest,
)

CORE_DRUGCENTRAL_TABLES = (
    "structures",
    "synonyms",
    "atc",
    "struct2atc",
    "drug_class",
)
_COPY_RE = re.compile(
    r'^COPY\s+(?:(?:"?public"?)\.)?"?(?P<table>[A-Za-z0-9_]+)"?\s*'
    r'\((?P<columns>[^)]+)\)\s+FROM\s+stdin;$',
    re.IGNORECASE,
)
_BATCH_ROWS = 50_000


class DrugCentralBuildError(RuntimeError):
    """Raised when the SQL dump cannot be converted safely."""


def build_drugcentral(
    *, data_dir: Path, tables: tuple[str, ...] = CORE_DRUGCENTRAL_TABLES
) -> list[ReferenceBuildRecord]:
    """Convert selected PostgreSQL COPY tables to individual Parquet files."""
    invalid = set(tables) - set(CORE_DRUGCENTRAL_TABLES)
    if invalid:
        raise ValueError(f"unsupported DrugCentral tables: {sorted(invalid)}")
    if len(tables) != len(set(tables)):
        raise ValueError("DrugCentral tables cannot contain duplicates")
    source = _find_dump(data_dir)
    checksum = sha256_file(source)
    output_dir = data_dir / "interim" / "drugcentral"
    records: list[ReferenceBuildRecord] = []
    pending: list[str] = []
    for table in tables:
        destination = output_dir / f"{table}.parquet"
        try:
            cached = cached_record(
                destination=destination,
                source=source,
                source_sha256=checksum,
                dataset="drugcentral",
                table=table,
            )
        except RuntimeError as exc:
            raise DrugCentralBuildError(str(exc)) from exc
        if cached is None:
            pending.append(table)
        else:
            records.append(cached)

    if pending:
        records.extend(
            _convert_copy_tables(
                source=source,
                source_sha256=checksum,
                output_dir=output_dir,
                tables=tuple(pending),
            )
        )
    records.sort(key=lambda record: record.table)
    write_manifest(
        output_dir / "manifest.json", dataset="DrugCentral interim Parquet", records=records
    )
    return records


def _find_dump(data_dir: Path) -> Path:
    directory = data_dir / "raw" / "drugcentral"
    matches = sorted(directory.glob("*.sql.gz"))
    if len(matches) != 1:
        raise DrugCentralBuildError(
            f"expected one DrugCentral .sql.gz dump in {directory}, found {len(matches)}; "
            "run `tekarx extract-drugcentral` first"
        )
    return matches[0]


def _convert_copy_tables(
    *, source: Path, source_sha256: str, output_dir: Path, tables: tuple[str, ...]
) -> list[ReferenceBuildRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(tables)
    found: set[str] = set()
    states: dict[str, dict[str, object]] = {}
    current: str | None = None
    try:
        with gzip.open(source, mode="rt", encoding="utf-8", newline="") as stream:
            progress = tqdm(desc="Scanning DrugCentral SQL", unit="lines", unit_scale=True)
            try:
                for line in stream:
                    progress.update()
                    if current is None:
                        match = _COPY_RE.match(line.rstrip("\r\n"))
                        if match is None:
                            continue
                        table = match.group("table").lower()
                        if table not in wanted:
                            continue
                        if table in found:
                            raise DrugCentralBuildError(f"duplicate COPY block for table {table}")
                        columns = [_identifier(item) for item in match.group("columns").split(",")]
                        if not columns or len(columns) != len(set(columns)):
                            raise DrugCentralBuildError(f"invalid columns for table {table}")
                        schema = pa.schema(
                            [pa.field(column, pa.string()) for column in columns],
                            metadata=_metadata(source, source_sha256, table),
                        )
                        temporary = (output_dir / f"{table}.parquet").with_suffix(".parquet.tmp")
                        temporary.unlink(missing_ok=True)
                        states[table] = {
                            "columns": columns,
                            "schema": schema,
                            "writer": pq.ParquetWriter(temporary, schema, compression="snappy"),
                            "temporary": temporary,
                            "buffers": [[] for _ in columns],
                            "rows": 0,
                        }
                        current = table
                        found.add(table)
                        continue
                    if line.rstrip("\r\n") == r"\.":
                        _flush(states[current])
                        writer = states[current]["writer"]
                        assert isinstance(writer, pq.ParquetWriter)
                        writer.close()
                        states[current]["writer"] = None
                        current = None
                        continue
                    _append_copy_row(states[current], line)
            finally:
                progress.close()
        if current is not None:
            raise DrugCentralBuildError(f"unterminated COPY block for table {current}")
        missing = wanted - found
        if missing:
            raise DrugCentralBuildError(f"tables not found in DrugCentral dump: {sorted(missing)}")

        records: list[ReferenceBuildRecord] = []
        for table in tables:
            state = states[table]
            temporary = state["temporary"]
            schema = state["schema"]
            rows = state["rows"]
            assert isinstance(temporary, Path)
            assert isinstance(schema, pa.Schema)
            assert isinstance(rows, int)
            verify_parquet(temporary, expected_rows=rows, expected_schema=schema)
            destination = output_dir / f"{table}.parquet"
            os.replace(temporary, destination)
            records.append(
                ReferenceBuildRecord(
                    dataset="drugcentral",
                    table=table,
                    source_path=str(source),
                    source_sha256=source_sha256,
                    output_path=str(destination),
                    rows=rows,
                    columns=len(schema),
                    output_size_bytes=destination.stat().st_size,
                )
            )
        return records
    except BaseException:
        for state in states.values():
            writer = state.get("writer")
            if isinstance(writer, pq.ParquetWriter):
                writer.close()
            temporary = state.get("temporary")
            if isinstance(temporary, Path):
                temporary.unlink(missing_ok=True)
        raise


def _identifier(value: str) -> str:
    return value.strip().strip('"').lower()


def _metadata(source: Path, checksum: str, table: str) -> dict[bytes, bytes]:
    return {
        b"tekarx.source_path": str(source).encode(),
        b"tekarx.source_sha256": checksum.encode(),
        b"tekarx.dataset": b"drugcentral",
        b"tekarx.table": table.encode(),
        b"tekarx.format": b"postgres-copy-text",
        b"tekarx.built_at_utc": datetime.now(UTC).isoformat().encode(),
    }


def _append_copy_row(state: dict[str, object], line: str) -> None:
    columns = state["columns"]
    buffers = state["buffers"]
    assert isinstance(columns, list)
    assert isinstance(buffers, list)
    values = line.rstrip("\r\n").split("\t")
    if len(values) != len(columns):
        raise DrugCentralBuildError(
            f"COPY row has {len(values)} fields; expected {len(columns)}"
        )
    for buffer, value in zip(buffers, values, strict=True):
        buffer.append(None if value == r"\N" else _unescape_copy(value))
    state["rows"] = int(state["rows"]) + 1
    if len(buffers[0]) >= _BATCH_ROWS:
        _flush(state)


def _flush(state: dict[str, object]) -> None:
    columns = state["columns"]
    buffers = state["buffers"]
    writer = state["writer"]
    assert isinstance(columns, list)
    assert isinstance(buffers, list)
    assert isinstance(writer, pq.ParquetWriter)
    if not buffers[0]:
        return
    batch = pa.record_batch(
        [pa.array(buffer, type=pa.string()) for buffer in buffers], names=columns
    )
    writer.write_batch(batch)
    for buffer in buffers:
        buffer.clear()


def _unescape_copy(value: str) -> str:
    """Decode PostgreSQL COPY text escapes without interpreting ordinary backslashes."""
    output: list[str] = []
    index = 0
    simple = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}
    while index < len(value):
        if value[index] != "\\" or index + 1 >= len(value):
            output.append(value[index])
            index += 1
            continue
        marker = value[index + 1]
        if marker in simple:
            output.append(simple[marker])
            index += 2
        elif marker in "01234567":
            end = index + 2
            while end < min(index + 4, len(value)) and value[end] in "01234567":
                end += 1
            output.append(chr(int(value[index + 1 : end], 8)))
            index = end
        elif marker == "x" and index + 2 < len(value):
            end = index + 2
            while end < min(index + 4, len(value)) and value[end] in "0123456789abcdefABCDEF":
                end += 1
            if end == index + 2:
                output.append("x")
                index += 2
            else:
                output.append(chr(int(value[index + 2 : end], 16)))
                index = end
        else:
            output.append(marker)
            index += 2
    return "".join(output)
