"""Stream official DailyMed mapping ZIPs to normalized Parquet tables."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from tekarx.extract.common import sha256_file
from tekarx.extract.dailymed import DAILYMED_DATASETS
from tekarx.transform.reference import (
    ReferenceBuildRecord,
    cached_record,
    verify_parquet,
    write_manifest,
)

DAILYMED_REQUIRED_COLUMNS = {
    "rxnorm": frozenset({"spl_setid", "spl_version", "rxcui", "rxstring", "rxtty"}),
    "pharmacologic-class": frozenset(
        {"spl_setid", "spl_version", "pharma_setid", "pharma_version"}
    ),
    "metadata": frozenset({"setid", "zip_file_name", "upload_date", "spl_version", "title"}),
}
_BATCH_ROWS = 50_000


class DailyMedBuildError(RuntimeError):
    """Raised when a DailyMed mapping archive cannot be converted safely."""


def build_dailymed(
    *, data_dir: Path, datasets: tuple[str, ...] = tuple(sorted(DAILYMED_DATASETS))
) -> list[ReferenceBuildRecord]:
    """Convert selected DailyMed pipe-delimited mapping archives to Parquet."""
    invalid = set(datasets) - set(DAILYMED_DATASETS)
    if invalid:
        raise ValueError(f"unsupported DailyMed datasets: {sorted(invalid)}")
    records = [_build_dataset(data_dir=data_dir, dataset=dataset) for dataset in datasets]
    write_manifest(
        data_dir / "interim" / "dailymed" / "manifest.json",
        dataset="DailyMed interim Parquet",
        records=records,
    )
    return records


def _build_dataset(*, data_dir: Path, dataset: str) -> ReferenceBuildRecord:
    _, filename = DAILYMED_DATASETS[dataset]
    source = data_dir / "raw" / "dailymed" / filename
    if not source.is_file():
        raise DailyMedBuildError(
            f"missing DailyMed {dataset} archive: {source}; run "
            f"`tekarx extract-dailymed --dataset {dataset}` first"
        )
    checksum = sha256_file(source)
    output_dir = data_dir / "interim" / "dailymed"
    destination = output_dir / f"{dataset}.parquet"
    try:
        cached = cached_record(
            destination=destination,
            source=source,
            source_sha256=checksum,
            dataset="dailymed",
            table=dataset,
        )
    except RuntimeError as exc:
        raise DailyMedBuildError(str(exc)) from exc
    if cached is not None:
        return cached
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, columns = _convert_archive(
        source=source,
        source_sha256=checksum,
        destination=destination,
        dataset=dataset,
    )
    return ReferenceBuildRecord(
        dataset="dailymed",
        table=dataset,
        source_path=str(source),
        source_sha256=checksum,
        output_path=str(destination),
        rows=rows,
        columns=columns,
        output_size_bytes=destination.stat().st_size,
    )


def _convert_archive(
    *, source: Path, source_sha256: str, destination: Path, dataset: str
) -> tuple[int, int]:
    temporary = destination.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    schema: pa.Schema | None = None
    member_names: list[str] = []
    try:
        with zipfile.ZipFile(source) as archive:
            members = _data_members(archive, dataset)
            if not members:
                raise DailyMedBuildError(f"no valid {dataset} mapping file found in {source}")
            member_names = [member.filename for member, _ in members]
            canonical_columns = members[0][1]
            if any(columns != canonical_columns for _, columns in members):
                raise DailyMedBuildError(f"inconsistent mapping headers in {source}")
            schema = pa.schema(
                [pa.field(column, pa.string()) for column in canonical_columns],
                metadata=_metadata(source, source_sha256, dataset, member_names),
            )
            writer = pq.ParquetWriter(temporary, schema, compression="snappy")
            progress = tqdm(desc=f"Building DailyMed {dataset}", unit="rows", unit_scale=True)
            try:
                for member, _ in members:
                    with archive.open(member) as binary:
                        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
                        reader = csv.reader(text, delimiter="|", quotechar='"')
                        next(reader)
                        buffers = [[] for _ in canonical_columns]
                        for values in reader:
                            if not values or all(not value for value in values):
                                continue
                            if len(values) != len(canonical_columns):
                                raise DailyMedBuildError(
                                    f"{member.filename} row has {len(values)} fields; "
                                    f"expected {len(canonical_columns)}"
                                )
                            for buffer, value in zip(buffers, values, strict=True):
                                buffer.append(value if value != "" else None)
                            rows += 1
                            if len(buffers[0]) >= _BATCH_ROWS:
                                _write_buffers(writer, buffers, canonical_columns)
                                progress.update(_BATCH_ROWS)
                        remaining = len(buffers[0])
                        _write_buffers(writer, buffers, canonical_columns)
                        progress.update(remaining)
            finally:
                progress.close()
        writer.close()
        writer = None
        assert schema is not None
        verify_parquet(temporary, expected_rows=rows, expected_schema=schema)
        os.replace(temporary, destination)
        return rows, len(schema)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise


def _data_members(
    archive: zipfile.ZipFile, dataset: str
) -> list[tuple[zipfile.ZipInfo, list[str]]]:
    matches: list[tuple[zipfile.ZipInfo, list[str]]] = []
    required = DAILYMED_REQUIRED_COLUMNS[dataset]
    for member in archive.infolist():
        if member.is_dir() or member.filename.startswith("__MACOSX/"):
            continue
        with archive.open(member) as binary:
            header = binary.readline().decode("utf-8-sig").rstrip("\r\n")
        columns = [_normalize_column(value) for value in next(csv.reader([header], delimiter="|"))]
        columns = _canonicalize_columns(columns, dataset=dataset)
        if required.issubset(columns):
            if not columns or len(columns) != len(set(columns)):
                raise DailyMedBuildError(f"invalid or duplicate header in {member.filename}")
            matches.append((member, columns))
    return matches


def _normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _canonicalize_columns(columns: list[str], *, dataset: str) -> list[str]:
    """Normalize known publisher header variants to a stable interim schema."""
    aliases = {"setid": "spl_setid"} if dataset in {"rxnorm", "pharmacologic-class"} else {}
    return [aliases.get(column, column) for column in columns]


def _write_buffers(
    writer: pq.ParquetWriter, buffers: list[list[str | None]], columns: list[str]
) -> None:
    if not buffers[0]:
        return
    writer.write_batch(
        pa.record_batch([pa.array(values, type=pa.string()) for values in buffers], names=columns)
    )
    for values in buffers:
        values.clear()


def _metadata(
    source: Path, checksum: str, dataset: str, members: list[str]
) -> dict[bytes, bytes]:
    return {
        b"tekarx.source_path": str(source).encode(),
        b"tekarx.source_sha256": checksum.encode(),
        b"tekarx.dataset": b"dailymed",
        b"tekarx.table": dataset.encode(),
        b"tekarx.archive_members": json.dumps(members).encode(),
        b"tekarx.format": b"zip-pipe-delimited",
        b"tekarx.built_at_utc": datetime.now(UTC).isoformat().encode(),
    }
