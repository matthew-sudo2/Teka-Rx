"""Build a leakage-safe, report-level FAERS cohort and graph edge tables."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from tekarx.extract.common import sha256_file
from tekarx.studies import FAERS_PRESETS

SERIOUS_OUTCOME_CODES = ("DE", "LT", "HO", "DS", "RI", "CA", "OT")
COHORT_BUILD_VERSION = 2


class CohortBuildError(RuntimeError):
    """Raised when staged FAERS data cannot produce a valid cohort."""


@dataclass(frozen=True)
class CohortBuildRecord:
    """Summary and provenance for a complete cohort build."""

    cohort_path: str
    case_splits_path: str
    drug_edges_path: str
    reaction_edges_path: str
    outcome_edges_path: str
    cohort_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    drug_edges: int
    reaction_edges: int
    outcome_edges: int
    input_files: int
    source_demo_rows: int
    source_cases: int
    deleted_cases: int
    nondeleted_cases: int
    latest_cases: int
    valid_age_cases: int
    cases_with_drugs: int
    cases_with_outcomes: int
    compression: str = "snappy"
    cached: bool = False


def build_cohort(
    *,
    data_dir: Path,
    split_preset: str = "gnn-small",
    memory_limit: str = "4GB",
    threads: int | None = None,
) -> CohortBuildRecord:
    """Build one latest report per case plus split-aware graph edge tables."""
    split_definition = _split_definition(split_preset)
    sources = _source_files(data_dir)
    source_checksums = {str(path): sha256_file(path) for path in sources}
    processed = data_dir / "processed"
    outputs = _output_paths(processed)
    manifest_path = processed / "cohort_manifest.json"
    cached = _cached_build(
        manifest_path=manifest_path,
        outputs=outputs,
        source_checksums=source_checksums,
        split_preset=split_preset,
        split_definition=split_definition,
    )
    if cached is not None:
        return cached

    processed.mkdir(parents=True, exist_ok=True)
    (processed / "edges").mkdir(parents=True, exist_ok=True)
    temporary_outputs = {name: path.with_suffix(".parquet.tmp") for name, path in outputs.items()}
    for path in temporary_outputs.values():
        path.unlink(missing_ok=True)

    database = data_dir / "interim" / f".cohort-build-{os.getpid()}.duckdb"
    database.unlink(missing_ok=True)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(database))
        connection.execute(f"SET memory_limit = '{_sql_literal(memory_limit)}'")
        if threads is not None:
            if threads < 1:
                raise ValueError("threads must be at least 1")
            connection.execute(f"SET threads = {threads}")
        temp_directory = data_dir / "interim" / ".duckdb_temp"
        temp_directory.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = '{_sql_literal(temp_directory.as_posix())}'")

        _create_source_views(connection, data_dir)
        _create_case_tables(connection, split_definition)
        _create_aggregates(connection)
        _create_cohort_table(connection)
        _create_edge_tables(connection)

        table_names = {
            "cohort": "cohort_final",
            "case_splits": "case_splits",
            "drug_edges": "report_drug_edges",
            "reaction_edges": "report_reaction_edges",
            "outcome_edges": "report_outcome_edges",
        }
        for name, table in table_names.items():
            _copy_parquet(connection, table=table, destination=temporary_outputs[name])

        stats = _audit_tables(connection)
        quarter_coverage = _quarter_coverage(connection, split_definition)
        empty_splits = [split for split in ("train", "validation", "test") if not stats[split]]
        if empty_splits:
            raise CohortBuildError(
                "cohort has no rows for required split(s): " + ", ".join(empty_splits)
            )
        connection.close()
        connection = None

        _warn_missing_quarters(quarter_coverage)

        for name, path in temporary_outputs.items():
            _verify_snappy(path, expected_rows=stats[name])
        for name, destination in outputs.items():
            os.replace(temporary_outputs[name], destination)

        record = CohortBuildRecord(
            cohort_path=str(outputs["cohort"]),
            case_splits_path=str(outputs["case_splits"]),
            drug_edges_path=str(outputs["drug_edges"]),
            reaction_edges_path=str(outputs["reaction_edges"]),
            outcome_edges_path=str(outputs["outcome_edges"]),
            cohort_rows=stats["cohort"],
            train_rows=stats["train"],
            validation_rows=stats["validation"],
            test_rows=stats["test"],
            drug_edges=stats["drug_edges"],
            reaction_edges=stats["reaction_edges"],
            outcome_edges=stats["outcome_edges"],
            input_files=len(sources),
            source_demo_rows=stats["source_demo_rows"],
            source_cases=stats["source_cases"],
            deleted_cases=stats["deleted_cases"],
            nondeleted_cases=stats["nondeleted_cases"],
            latest_cases=stats["latest_cases"],
            valid_age_cases=stats["valid_age_cases"],
            cases_with_drugs=stats["cases_with_drugs"],
            cases_with_outcomes=stats["cases_with_outcomes"],
        )
        _write_manifest(
            manifest_path,
            record=record,
            source_checksums=source_checksums,
            split_preset=split_preset,
            split_definition=split_definition,
            quarter_coverage=quarter_coverage,
            memory_limit=memory_limit,
        )
        return record
    except BaseException:
        if connection is not None:
            connection.close()
        for path in temporary_outputs.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        database.unlink(missing_ok=True)


def _split_definition(split_preset: str) -> dict[str, tuple[str, ...]]:
    try:
        configured = FAERS_PRESETS[split_preset]
    except KeyError as exc:
        choices = ", ".join(sorted(FAERS_PRESETS))
        raise ValueError(f"unknown split preset {split_preset!r}; choose: {choices}") from exc

    required = {"train", "validation", "test"}
    if set(configured) != required:
        raise CohortBuildError(
            f"split preset {split_preset!r} must define exactly train, validation, and test"
        )
    normalized = {
        split: tuple(str(quarter).strip().upper() for quarter in configured[split])
        for split in ("train", "validation", "test")
    }
    if any(not quarters for quarters in normalized.values()):
        raise CohortBuildError(f"split preset {split_preset!r} contains an empty split")
    all_quarters = [quarter for quarters in normalized.values() for quarter in quarters]
    if len(all_quarters) != len(set(all_quarters)):
        raise CohortBuildError(f"split preset {split_preset!r} assigns a quarter more than once")
    return normalized


def _source_files(data_dir: Path) -> list[Path]:
    root = data_dir / "interim" / "faers"
    files: list[Path] = []
    for table in ("demo", "drug", "reac", "outc", "delete"):
        table_files = sorted((root / table).glob("*.parquet"))
        if not table_files:
            raise CohortBuildError(
                f"missing staged FAERS {table} Parquet files; run `tekarx build-faers` first"
            )
        files.extend(table_files)
    return files


def _output_paths(processed: Path) -> dict[str, Path]:
    return {
        "cohort": processed / "tekarx_cohort.parquet",
        "case_splits": processed / "case_splits.parquet",
        "drug_edges": processed / "edges" / "report_drug.parquet",
        "reaction_edges": processed / "edges" / "report_reaction.parquet",
        "outcome_edges": processed / "edges" / "report_outcome.parquet",
    }


def _create_source_views(connection: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    root = data_dir / "interim" / "faers"
    for table in ("demo", "drug", "reac", "outc", "delete"):
        glob = _sql_literal((root / table / "*.parquet").as_posix())
        connection.execute(
            f"CREATE VIEW {table}_source AS "
            f"SELECT * FROM read_parquet('{glob}', union_by_name = true)"
        )


def _report_date_sql(alias: str = "d") -> str:
    quarter_end = (
        f"substr({alias}.quarter, 1, 4) || CASE right({alias}.quarter, 1) "
        "WHEN '1' THEN '0331' WHEN '2' THEN '0630' "
        "WHEN '3' THEN '0930' ELSE '1231' END"
    )
    return (
        f"COALESCE(try_strptime({alias}.fda_dt, '%Y%m%d')::DATE, "
        f"try_strptime({alias}.rept_dt, '%Y%m%d')::DATE, "
        f"try_strptime({alias}.event_dt, '%Y%m%d')::DATE, "
        f"try_strptime({quarter_end}, '%Y%m%d')::DATE)"
    )


def _age_years_sql(alias: str = "d") -> str:
    age = f"try_cast({alias}.age AS DOUBLE)"
    unit = f"upper(trim({alias}.age_cod))"
    return (
        f"CASE WHEN {unit} IN ('YR', 'YEAR', 'YEARS') THEN {age} "
        f"WHEN {unit} IN ('MON', 'MO', 'MONTH', 'MONTHS') THEN {age} / 12.0 "
        f"WHEN {unit} IN ('WK', 'WEEK', 'WEEKS') THEN {age} / 52.142857 "
        f"WHEN {unit} IN ('DY', 'DAY', 'DAYS') THEN {age} / 365.25 "
        f"WHEN {unit} IN ('HR', 'HOUR', 'HOURS') THEN {age} / 8766.0 "
        f"WHEN {unit} IN ('DEC', 'DECADE', 'DECADES') THEN {age} * 10.0 "
        "ELSE NULL END"
    )


def _create_case_tables(
    connection: duckdb.DuckDBPyConnection,
    split_definition: dict[str, tuple[str, ...]],
) -> None:
    report_date = _report_date_sql()
    age_years = _age_years_sql()
    connection.execute(
        "CREATE TABLE deleted_cases AS "
        "SELECT DISTINCT caseid FROM delete_source WHERE caseid IS NOT NULL"
    )
    connection.execute(
        f"""
        CREATE TABLE demo_clean AS
        SELECT d.*, {report_date} AS report_date,
               try_cast(d.caseversion AS BIGINT) AS caseversion_num,
               {age_years} AS age_years
        FROM demo_source d
        WHERE d.caseid IS NOT NULL
          AND d.primaryid IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM deleted_cases x WHERE x.caseid = d.caseid
          )
        """
    )
    connection.execute(
        """
        CREATE TABLE case_dates AS
        SELECT caseid, max(report_date) AS max_report_date
        FROM demo_clean
        GROUP BY caseid
        """
    )
    split_rows = ", ".join(
        f"('{_sql_literal(quarter)}', '{_sql_literal(split)}')"
        for split, quarters in split_definition.items()
        for quarter in quarters
    )
    connection.execute("CREATE TEMP TABLE split_quarters(quarter VARCHAR, split VARCHAR)")
    connection.execute(f"INSERT INTO split_quarters VALUES {split_rows}")
    connection.execute(
        """
        CREATE TABLE latest_demo_all AS
        WITH ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY caseid
                ORDER BY caseversion_num DESC NULLS LAST,
                         report_date DESC NULLS LAST,
                         try_cast(primaryid AS HUGEINT) DESC NULLS LAST,
                         primaryid DESC
            ) AS version_rank
            FROM demo_clean
        )
        SELECT primaryid, caseid, caseversion_num AS caseversion,
               report_date, upper(trim(quarter)) AS quarter, age_years, sex, wt, wt_cod
        FROM ranked
        WHERE version_rank = 1
        """
    )
    connection.execute(
        """
        CREATE TABLE case_splits AS
        SELECT l.caseid, l.report_date AS max_report_date, l.quarter, q.split
        FROM latest_demo_all l
        JOIN split_quarters q USING (quarter)
        """
    )
    connection.execute(
        "CREATE TABLE latest_demo_age_valid AS SELECT * FROM latest_demo_all "
        "WHERE age_years IS NULL OR age_years BETWEEN 0 AND 120"
    )
    connection.execute(
        "CREATE TABLE latest_demo AS "
        "SELECT l.* FROM latest_demo_age_valid l JOIN case_splits s USING (caseid)"
    )


def _create_aggregates(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE drug_agg AS
        WITH cleaned AS (
            SELECT d.primaryid, upper(trim(d.drugname)) AS drugname
            FROM drug_source d
            JOIN latest_demo l USING (primaryid)
            WHERE d.drugname IS NOT NULL AND trim(d.drugname) <> ''
        )
        SELECT primaryid,
               string_agg(DISTINCT drugname, '|' ORDER BY drugname) AS drug_list_str,
               count(DISTINCT drugname)::INTEGER AS num_drugs
        FROM cleaned
        GROUP BY primaryid
        """
    )
    connection.execute(
        """
        CREATE TABLE reaction_agg AS
        WITH cleaned AS (
            SELECT r.primaryid, trim(r.pt) AS reaction
            FROM reac_source r
            JOIN latest_demo l USING (primaryid)
            WHERE r.pt IS NOT NULL AND trim(r.pt) <> ''
        )
        SELECT primaryid,
               string_agg(DISTINCT reaction, '|' ORDER BY reaction) AS reaction_list_str,
               count(DISTINCT reaction)::INTEGER AS num_reactions
        FROM cleaned
        GROUP BY primaryid
        """
    )
    serious = ", ".join(f"'{code}'" for code in SERIOUS_OUTCOME_CODES)
    connection.execute(
        f"""
        CREATE TABLE outcome_agg AS
        WITH cleaned AS (
            SELECT o.primaryid, upper(trim(o.outc_cod)) AS outcome
            FROM outc_source o
            JOIN latest_demo l USING (primaryid)
            WHERE o.outc_cod IS NOT NULL AND trim(o.outc_cod) <> ''
        )
        SELECT primaryid,
               string_agg(DISTINCT outcome, '|' ORDER BY outcome) AS outcome_codes,
               max(CASE WHEN outcome IN ({serious}) THEN 1 ELSE 0 END)::INTEGER AS is_serious
        FROM cleaned
        GROUP BY primaryid
        """
    )


def _create_cohort_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE cohort_final AS
        SELECT l.primaryid,
               l.caseid,
               l.caseversion,
               l.report_date,
               l.quarter,
               floor(l.age_years)::INTEGER AS age,
               l.age_years,
               CASE WHEN upper(trim(l.sex)) IN ('M', 'F')
                    THEN upper(trim(l.sex)) ELSE 'UNKNOWN' END AS sex,
               CASE WHEN upper(trim(l.sex)) IN ('M', 'F')
                    THEN 0 ELSE 1 END::INTEGER AS sex_unknown,
               CASE WHEN l.age_years IS NULL THEN 1 ELSE 0 END::INTEGER AS age_missing,
               l.wt AS weight,
               l.wt_cod AS weight_unit,
               CASE
                   WHEN l.age_years IS NULL THEN 'UNKNOWN'
                   WHEN l.age_years < 18 THEN '0-17'
                   WHEN l.age_years < 41 THEN '18-40'
                   WHEN l.age_years < 65 THEN '41-64'
                   ELSE '65+'
               END AS age_group,
               d.drug_list_str,
               d.num_drugs,
               CASE WHEN d.num_drugs >= 5 THEN 1 ELSE 0 END::INTEGER AS is_polypharmacy,
               r.reaction_list_str,
               coalesce(r.num_reactions, 0)::INTEGER AS num_reactions,
               o.outcome_codes,
               coalesce(o.is_serious, 0)::INTEGER AS is_serious,
               s.split
        FROM latest_demo l
        JOIN drug_agg d USING (primaryid)
        LEFT JOIN outcome_agg o USING (primaryid)
        LEFT JOIN reaction_agg r USING (primaryid)
        JOIN case_splits s USING (caseid)
        ORDER BY l.report_date, l.caseid
        """
    )


def _create_edge_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE report_drug_edges AS
        SELECT DISTINCT c.primaryid, c.caseid, c.split,
               d.drug_seq, d.role_cod, d.drugname, d.prod_ai,
               d.route, d.dose_amt, d.dose_unit, d.dose_form, d.dose_freq
        FROM cohort_final c
        JOIN drug_source d USING (primaryid, caseid)
        WHERE d.drugname IS NOT NULL AND trim(d.drugname) <> ''
        """
    )
    connection.execute(
        """
        CREATE TABLE report_reaction_edges AS
        SELECT DISTINCT c.primaryid, c.caseid, c.split, r.pt, r.drug_rec_act
        FROM cohort_final c
        JOIN reac_source r USING (primaryid, caseid)
        WHERE r.pt IS NOT NULL AND trim(r.pt) <> ''
        """
    )
    connection.execute(
        """
        CREATE TABLE report_outcome_edges AS
        SELECT DISTINCT c.primaryid, c.caseid, c.split, upper(trim(o.outc_cod)) AS outc_cod
        FROM cohort_final c
        JOIN outc_source o USING (primaryid, caseid)
        WHERE o.outc_cod IS NOT NULL AND trim(o.outc_cod) <> ''
        """
    )


def _copy_parquet(connection: duckdb.DuckDBPyConnection, *, table: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = _sql_literal(destination.as_posix())
    connection.execute(
        f"COPY (SELECT * FROM {table}) TO '{path}' "
        "(FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000)"
    )


def _audit_tables(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    duplicate_reports = connection.execute(
        "SELECT count(*) FROM (SELECT primaryid FROM cohort_final "
        "GROUP BY primaryid HAVING count(*) > 1)"
    ).fetchone()[0]
    duplicate_cases = connection.execute(
        "SELECT count(*) FROM (SELECT caseid FROM cohort_final GROUP BY caseid HAVING count(*) > 1)"
    ).fetchone()[0]
    split_leakage = connection.execute(
        "SELECT count(*) FROM (SELECT caseid FROM case_splits GROUP BY caseid "
        "HAVING count(DISTINCT split) > 1)"
    ).fetchone()[0]
    if duplicate_reports or duplicate_cases or split_leakage:
        raise CohortBuildError(
            "cohort audit failed: "
            f"duplicate_reports={duplicate_reports}, duplicate_cases={duplicate_cases}, "
            f"split_leakage={split_leakage}"
        )
    result = {
        "cohort": _count(connection, "cohort_final"),
        "case_splits": _count(connection, "case_splits"),
        "drug_edges": _count(connection, "report_drug_edges"),
        "reaction_edges": _count(connection, "report_reaction_edges"),
        "outcome_edges": _count(connection, "report_outcome_edges"),
        "source_demo_rows": _count(connection, "demo_source"),
        "source_cases": connection.execute(
            "SELECT count(DISTINCT caseid) FROM demo_source WHERE caseid IS NOT NULL"
        ).fetchone()[0],
        "deleted_cases": _count(connection, "deleted_cases"),
        "nondeleted_cases": _count(connection, "case_dates"),
        "latest_cases": _count(connection, "latest_demo_all"),
        "valid_age_cases": _count(connection, "latest_demo_age_valid"),
        "cases_with_drugs": _count(connection, "drug_agg"),
        "cases_with_outcomes": _count(connection, "outcome_agg"),
    }
    for split in ("train", "validation", "test"):
        result[split] = connection.execute(
            "SELECT count(*) FROM cohort_final WHERE split = ?", [split]
        ).fetchone()[0]
    if sum(result[split] for split in ("train", "validation", "test")) != result["cohort"]:
        raise CohortBuildError("split row counts do not sum to the cohort row count")
    return result


def _count(connection: duckdb.DuckDBPyConnection, table: str) -> int:
    return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _quarter_coverage(
    connection: duckdb.DuckDBPyConnection,
    split_definition: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, list[str]]]:
    staged = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT upper(trim(quarter)) FROM demo_source "
            "WHERE quarter IS NOT NULL AND trim(quarter) <> ''"
        ).fetchall()
    }
    present = {
        split: [quarter for quarter in quarters if quarter in staged]
        for split, quarters in split_definition.items()
    }
    missing = {
        split: [quarter for quarter in quarters if quarter not in staged]
        for split, quarters in split_definition.items()
    }
    return {"present": present, "missing": missing}


def _warn_missing_quarters(quarter_coverage: dict[str, dict[str, list[str]]]) -> None:
    missing_quarters = [
        quarter for quarters in quarter_coverage["missing"].values() for quarter in quarters
    ]
    if missing_quarters:
        print(
            "Warning: staged FAERS data is missing preset quarter(s): "
            + ", ".join(missing_quarters)
            + ". The manifest records the actual coverage.",
            file=sys.stderr,
        )


def _verify_snappy(path: Path, *, expected_rows: int) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise CohortBuildError(f"row-count mismatch for {path}")
    for row_group in range(parquet.metadata.num_row_groups):
        for column in range(parquet.metadata.num_columns):
            if parquet.metadata.row_group(row_group).column(column).compression != "SNAPPY":
                raise CohortBuildError(f"non-Snappy column found in {path}")


def _cached_build(
    *,
    manifest_path: Path,
    outputs: dict[str, Path],
    source_checksums: dict[str, str],
    split_preset: str,
    split_definition: dict[str, tuple[str, ...]],
) -> CohortBuildRecord | None:
    if not manifest_path.is_file() or not all(path.is_file() for path in outputs.values()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        manifest.get("build_version") != COHORT_BUILD_VERSION
        or manifest.get("input_sha256") != source_checksums
        or manifest.get("split_preset") != split_preset
        or manifest.get("splits")
        != {split: list(quarters) for split, quarters in split_definition.items()}
    ):
        return None
    record_data = manifest.get("record")
    if not isinstance(record_data, dict):
        return None
    record_data = dict(record_data)
    record_data["cached"] = True
    try:
        record = CohortBuildRecord(**record_data)
        expected = {
            "cohort": record.cohort_rows,
            "case_splits": pq.ParquetFile(outputs["case_splits"]).metadata.num_rows,
            "drug_edges": record.drug_edges,
            "reaction_edges": record.reaction_edges,
            "outcome_edges": record.outcome_edges,
        }
        for name, path in outputs.items():
            _verify_snappy(path, expected_rows=expected[name])
    except (TypeError, OSError, CohortBuildError):
        return None
    coverage = manifest.get("quarter_coverage")
    if isinstance(coverage, dict) and set(coverage) == {"present", "missing"}:
        _warn_missing_quarters(coverage)
    return record


def _write_manifest(
    path: Path,
    *,
    record: CohortBuildRecord,
    source_checksums: dict[str, str],
    split_preset: str,
    split_definition: dict[str, tuple[str, ...]],
    quarter_coverage: dict[str, dict[str, list[str]]],
    memory_limit: str,
) -> None:
    payload = {
        "dataset": "TekaRx FAERS report cohort",
        "build_version": COHORT_BUILD_VERSION,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "latest_version_key": "caseid/caseversion",
        "split_strategy": "case-level exact source-quarter temporal holdout",
        "split_preset": split_preset,
        "splits": {split: list(quarters) for split, quarters in split_definition.items()},
        "quarter_coverage": quarter_coverage,
        "memory_limit": memory_limit,
        "input_sha256": source_checksums,
        "serious_outcome_codes": list(SERIOUS_OUTCOME_CODES),
        "target_policy": (
            "is_serious=1 for any official FAERS outcome code; reports without an outcome "
            "row are retained with is_serious=0"
        ),
        "age_policy": (
            "normalize supported units to years; retain missing/unconvertible age with "
            "age_missing=1; exclude only normalized ages outside [0, 120]"
        ),
        "sex_policy": "normalize M/F and encode all other or missing values as UNKNOWN",
        "record": asdict(record),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")
