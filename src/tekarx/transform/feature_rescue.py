"""Build train-frozen FAERS feature-rescue lookups and retrain the baseline."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from tekarx.extract.common import sha256_file
from tekarx.transform.duckdb_runtime import configure_duckdb
from tekarx.transform.graph import GraphBuildRecord, build_graph

INDICATION_HASH_BINS = 32
PAIR_HIGH_RISK_ROR = 1.5
FEATURE_RESCUE_BUCKETS = 64
PAIR_RISK_SCOPE = (
    "strict-prior-date temporal encoding for fit-train patients; frozen full-fit-train "
    "lookup for later train/validation/test patients"
)

_PATIENT_BUCKET = "__tekarx_patient_bucket"
_PAIR_BUCKET = "__tekarx_pair_bucket"

MALIGNANCY_PATTERN = (
    r"MALIGN|CANCER|CARCINOMA|NEOPLASM|TUMOU?R|LEUKA?EMIA|LYMPHOMA|MYELOMA|"
    r"SARCOMA|MELANOMA|METAST"
)
CARDIO_PATTERN = (
    r"CARDI|HEART|HYPERTENS|CORONARY|ARRHYTH|ATRIAL|MYOCARD|ANGINA|STROKE|"
    r"THROMB|EMBOL|HYPERLIP|DYSLIP"
)
INFECTION_PATTERN = (
    r"INFECT|PNEUMON|SEPSIS|SEPTIC|VIRAL|BACTERI|FUNG|COVID|INFLUENZA|"
    r"HEPATITIS|\bHIV\b|TUBERCUL|ABSCESS"
)


class FeatureRescueError(RuntimeError):
    """Raised when feature-rescue inputs or temporal invariants are invalid."""


@dataclass(frozen=True)
class FeatureRescueRecord:
    """Summary for one train-frozen feature-rescue build."""

    output_path: str
    indication_lookup_path: str
    pair_lookup_path: str
    rows: int
    top_pairs: int
    training_reports: int
    training_min_date: str
    training_max_date: str
    pair_risk_scope: str
    indication_hash_bins: int
    patient_feature_count: int | None
    validation_auc: float | None
    graph_path: str | None


def build_feature_rescue(
    *,
    data_dir: Path,
    training_end_year: int = 2023,
    top_pairs: int = 50,
    minimum_pair_reports: int = 25,
    memory_limit: str = "4GB",
    threads: int | None = None,
    rebuild_graph: bool = True,
) -> FeatureRescueRecord:
    """Fit train-only indication/pair lookups, enrich patients, and rebuild models."""
    if top_pairs < 1 or minimum_pair_reports < 1:
        raise ValueError("top_pairs and minimum_pair_reports must be positive")
    root = Path(data_dir)
    processed = root / "processed"
    base = processed / "tekarx_cohort_enriched.parquet"
    if not base.is_file():
        base = processed / "tekarx_cohort.parquet"
    dictionary = processed / "drug_dictionary.parquet"
    indi_files = sorted((root / "interim" / "faers" / "indi").glob("*.parquet"))
    if not base.is_file() or not dictionary.is_file():
        raise FeatureRescueError("missing cohort or drug dictionary inputs")
    if not indi_files:
        raise FeatureRescueError(
            "missing staged FAERS INDI tables; run `tekarx build-faers --preset gnn-small "
            "--tables indi`"
        )

    processed.mkdir(parents=True, exist_ok=True)
    output = processed / "tekarx_cohort_feature_rescue.parquet"
    pair_output = processed / "high_risk_drug_pairs.parquet"
    indication_output = processed / "indication_lookup.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    pair_temporary = pair_output.with_suffix(".parquet.tmp")
    indication_temporary = indication_output.with_suffix(".parquet.tmp")
    for path in (temporary, pair_temporary, indication_temporary):
        path.unlink(missing_ok=True)

    interim = root / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".feature-rescue-work-", dir=interim))
    database = work_dir / "feature-rescue.duckdb"
    connection: duckdb.DuckDBPyConnection | None = None
    duckdb_temporary: Path | None = None
    expected_rows = pq.ParquetFile(base).metadata.num_rows
    try:
        connection = duckdb.connect(str(database))
        stage_name = work_dir.name.removeprefix(".").replace(".", "-")
        duckdb_temporary = configure_duckdb(
            connection,
            data_dir=root,
            stage=stage_name,
            memory_limit=memory_limit,
            threads=threads,
        )
        coverage = _build_rescue_tables(
            connection,
            base=base,
            dictionary=dictionary,
            indi_files=indi_files,
            training_end_year=training_end_year,
            top_pairs=top_pairs,
            minimum_pair_reports=minimum_pair_reports,
            work_dir=work_dir,
            buckets=FEATURE_RESCUE_BUCKETS,
        )
        _copy_table(connection, "top_pair_lookup", pair_temporary)
        _copy_table(connection, "indication_lookup", indication_temporary)
        print("Feature rescue [6/7]: assembling patient buckets", flush=True)
        _write_bucketed_enriched_cohort(
            connection,
            work_dir=work_dir,
            destination=temporary,
        )
        print("Feature rescue [7/7]: auditing atomic outputs", flush=True)
    except BaseException:
        for path in (temporary, pair_temporary, indication_temporary):
            path.unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:
            connection.close()
        shutil.rmtree(work_dir, ignore_errors=True)
        if duckdb_temporary is not None:
            shutil.rmtree(duckdb_temporary, ignore_errors=True)

    _verify_output(temporary, expected_rows=expected_rows)
    _verify_snappy(pair_temporary, label="high-risk pair lookup")
    _verify_snappy(indication_temporary, label="indication lookup")
    os.replace(temporary, output)
    os.replace(pair_temporary, pair_output)
    os.replace(indication_temporary, indication_output)

    graph_record: GraphBuildRecord | None = None
    if rebuild_graph:
        graph_record = build_graph(
            data_dir=root,
            cohort_path=output,
            memory_limit=memory_limit,
            threads=threads,
            xgb_rounds=1000,
            xgb_early_stopping=50,
            xgb_max_depth=0,
            xgb_max_leaves=63,
        )
    record = FeatureRescueRecord(
        output_path=str(output),
        indication_lookup_path=str(indication_output),
        pair_lookup_path=str(pair_output),
        rows=expected_rows,
        top_pairs=pq.ParquetFile(pair_output).metadata.num_rows,
        training_reports=int(coverage[0]),
        training_min_date=str(coverage[1]),
        training_max_date=str(coverage[2]),
        pair_risk_scope=PAIR_RISK_SCOPE,
        indication_hash_bins=INDICATION_HASH_BINS,
        patient_feature_count=graph_record.patient_feature_count if graph_record else None,
        validation_auc=graph_record.validation_auc if graph_record else None,
        graph_path=graph_record.graph_path if graph_record else None,
    )
    _write_manifest(
        processed / "feature_rescue_manifest.json",
        record=record,
        input_checksums={str(base): sha256_file(base), str(dictionary): sha256_file(dictionary)},
        training_end_year=training_end_year,
        minimum_pair_reports=minimum_pair_reports,
    )
    return record


def _build_rescue_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    base: Path,
    dictionary: Path,
    indi_files: list[Path],
    training_end_year: int,
    top_pairs: int,
    minimum_pair_reports: int,
    work_dir: Path,
    buckets: int,
) -> tuple[object, object, object]:
    if buckets < 1:
        raise ValueError("feature-rescue buckets must be positive")
    base_sql = _sql_literal(base.as_posix())
    coverage = connection.execute(
        f"""
        SELECT count(*), min(report_date), max(report_date),
               count(*) FILTER (WHERE is_serious = 1),
               count(*) FILTER (WHERE is_serious = 0)
        FROM read_parquet('{base_sql}')
        WHERE split = 'train' AND year(report_date) <= {int(training_end_year)}
        """
    ).fetchone()
    if not coverage or not coverage[0]:
        raise FeatureRescueError("no training reports satisfy the temporal cutoff")

    print(
        f"Feature rescue [1/7]: partitioning {coverage[0]:,} fit-train reports "
        f"and the cohort into {buckets} patient buckets",
        flush=True,
    )
    _partition_base(connection, base=base, work_dir=work_dir, buckets=buckets)
    _create_training_total_history(
        connection,
        base=base,
        training_end_year=training_end_year,
    )

    print("Feature rescue [2/7]: generating unique ingredient pairs", flush=True)
    _partition_patient_pairs(
        connection,
        base=base,
        dictionary=dictionary,
        work_dir=work_dir,
        training_end_year=training_end_year,
        buckets=buckets,
    )

    print("Feature rescue [3/7]: fitting and applying pair risk by pair bucket", flush=True)
    _build_pair_features(
        connection,
        work_dir=work_dir,
        total_serious=int(coverage[3]),
        total_nonserious=int(coverage[4]),
        minimum_pair_reports=minimum_pair_reports,
        buckets=buckets,
    )
    _create_top_pair_lookup(connection, work_dir=work_dir, top_pairs=top_pairs)

    print("Feature rescue [4/7]: reducing pair partials by patient bucket", flush=True)
    _reduce_pair_partials(connection, work_dir=work_dir, buckets=buckets)

    print("Feature rescue [5/7]: fitting train-only indication vocabulary", flush=True)
    _build_indication_features(
        connection,
        indi_files=indi_files,
        work_dir=work_dir,
        training_end_year=training_end_year,
        buckets=buckets,
    )
    return coverage[:3]


def _partition_base(
    connection: duckdb.DuckDBPyConnection,
    *,
    base: Path,
    work_dir: Path,
    buckets: int,
) -> None:
    destination = work_dir / "base-buckets"
    destination.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
            SELECT c.*,
                   cast(hash(primaryid) % {buckets} AS INTEGER) AS {_PATIENT_BUCKET}
            FROM read_parquet('{_sql_literal(base.as_posix())}') c
        ) TO '{_sql_literal(destination.as_posix())}' (
            FORMAT PARQUET,
            COMPRESSION SNAPPY,
            PARTITION_BY ({_PATIENT_BUCKET}),
            ROW_GROUP_SIZE 100000
        )
        """
    )


def _create_training_total_history(
    connection: duckdb.DuckDBPyConnection,
    *,
    base: Path,
    training_end_year: int,
) -> None:
    connection.execute(
        f"""
        CREATE TABLE training_total_history AS
        WITH daily AS (
            SELECT report_date,
                   count(*) FILTER (WHERE is_serious = 1)::BIGINT AS serious_on_date,
                   count(*) FILTER (WHERE is_serious = 0)::BIGINT AS nonserious_on_date
            FROM read_parquet('{_sql_literal(base.as_posix())}')
            WHERE split = 'train'
              AND year(report_date) <= {int(training_end_year)}
              AND report_date IS NOT NULL
            GROUP BY report_date
        )
        SELECT report_date,
               sum(serious_on_date) OVER (
                   ORDER BY report_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS total_serious,
               sum(nonserious_on_date) OVER (
                   ORDER BY report_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS total_nonserious
        FROM daily
        """
    )


def _partition_patient_pairs(
    connection: duckdb.DuckDBPyConnection,
    *,
    base: Path,
    dictionary: Path,
    work_dir: Path,
    training_end_year: int,
    buckets: int,
) -> None:
    drug_source = work_dir / "patient-drug-source"
    pair_source = work_dir / "pair-exposures"
    drug_source.mkdir(parents=True, exist_ok=True)
    pair_source.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
            WITH raw_drugs AS (
                SELECT c.primaryid, trim(value) AS faers_raw
                FROM read_parquet('{_sql_literal(base.as_posix())}') c,
                     unnest(string_split(c.drug_list_str, '|')) names(value)
                WHERE trim(value) <> ''
            )
            SELECT r.primaryid, d.dc_id::BIGINT AS dc_id,
                   cast(hash(r.primaryid) % {buckets} AS INTEGER) AS {_PATIENT_BUCKET}
            FROM raw_drugs r
            JOIN read_parquet('{_sql_literal(dictionary.as_posix())}') d USING (faers_raw)
            WHERE d.dc_id > 0
        ) TO '{_sql_literal(drug_source.as_posix())}' (
            FORMAT PARQUET,
            COMPRESSION SNAPPY,
            PARTITION_BY ({_PATIENT_BUCKET}),
            ROW_GROUP_SIZE 100000
        )
        """
    )

    patient_buckets = _partition_values(drug_source, _PATIENT_BUCKET)
    started = time.monotonic()
    for completed, bucket in enumerate(patient_buckets, start=1):
        drug_files = _partition_files(drug_source, _PATIENT_BUCKET, bucket)
        base_files = _partition_files(work_dir / "base-buckets", _PATIENT_BUCKET, bucket)
        if not drug_files or not base_files:
            continue
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE bucket_patient_drugs AS
            SELECT DISTINCT primaryid, dc_id
            FROM {_read_parquet_sql(drug_files)}
            """
        )
        bucket_destination = pair_source / f"patient_bucket={bucket}"
        bucket_destination.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"""
            COPY (
                SELECT a.primaryid, a.dc_id AS dc_id_1, b.dc_id AS dc_id_2,
                       c.report_date, c.is_serious,
                       coalesce((
                           c.split = 'train'
                           AND year(c.report_date) <= {int(training_end_year)}
                       ), false)::INTEGER AS is_fit_train,
                       cast(hash(a.dc_id, b.dc_id) % {buckets} AS INTEGER)
                           AS {_PAIR_BUCKET}
                FROM bucket_patient_drugs a
                JOIN bucket_patient_drugs b
                  ON a.primaryid = b.primaryid AND a.dc_id < b.dc_id
                JOIN {_read_parquet_sql(base_files)} c ON c.primaryid = a.primaryid
            ) TO '{_sql_literal(bucket_destination.as_posix())}' (
                FORMAT PARQUET,
                COMPRESSION SNAPPY,
                PARTITION_BY ({_PAIR_BUCKET}),
                ROW_GROUP_SIZE 100000
            )
            """
        )
        connection.execute("DROP TABLE IF EXISTS bucket_patient_drugs")
        _delete_files(drug_files)
        if completed == 1 or completed % 8 == 0 or completed == len(patient_buckets):
            print(
                f"  unique pairs: patient bucket {completed}/{len(patient_buckets)} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )
    shutil.rmtree(drug_source, ignore_errors=True)


def _build_pair_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    work_dir: Path,
    total_serious: int,
    total_nonserious: int,
    minimum_pair_reports: int,
    buckets: int,
) -> None:
    pair_source = work_dir / "pair-exposures"
    risk_parts = work_dir / "pair-risk-parts"
    partials = work_dir / "pair-patient-partials"
    risk_parts.mkdir(parents=True, exist_ok=True)
    partials.mkdir(parents=True, exist_ok=True)
    pair_buckets = _nested_partition_values(pair_source, _PAIR_BUCKET)
    started = time.monotonic()
    for completed, pair_bucket in enumerate(pair_buckets, start=1):
        pair_files = sorted(
            path
            for path in pair_source.glob(
                f"patient_bucket=*/{_PAIR_BUCKET}={pair_bucket}/*.parquet"
            )
            if path.is_file()
        )
        if not pair_files:
            continue
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE pair_exposures AS
            SELECT primaryid, dc_id_1, dc_id_2, report_date, is_serious, is_fit_train
            FROM {_read_parquet_sql(pair_files)}
            """
        )
        _delete_files(pair_files)
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE eligible_pair_risk AS
            WITH counts AS (
                SELECT dc_id_1, dc_id_2,
                       count(*) FILTER (
                           WHERE is_fit_train = 1 AND is_serious = 1
                       )::BIGINT AS a,
                       count(*) FILTER (
                           WHERE is_fit_train = 1 AND is_serious = 0
                       )::BIGINT AS b
                FROM pair_exposures
                WHERE is_fit_train = 1
                GROUP BY dc_id_1, dc_id_2
            )
            SELECT dc_id_1, dc_id_2, a, b,
                   ({total_serious} - a)::BIGINT AS c,
                   ({total_nonserious} - b)::BIGINT AS d,
                   (a + b)::BIGINT AS pair_reports,
                   (a::DOUBLE / nullif(a + b, 0)) /
                   nullif(
                       ({total_serious} - a)::DOUBLE /
                       nullif({total_serious + total_nonserious} - a - b, 0),
                       0
                   ) AS prr,
                   ln(
                       ((a + 0.5)::DOUBLE * ({total_nonserious} - b + 0.5)::DOUBLE) /
                       ((b + 0.5)::DOUBLE * ({total_serious} - a + 0.5)::DOUBLE)
                   )::DOUBLE AS log_ror
            FROM counts
            WHERE a + b >= {int(minimum_pair_reports)}
              AND {total_serious} > 0 AND {total_nonserious} > 0
            """
        )
        eligible_count = int(
            connection.execute("SELECT count(*) FROM eligible_pair_risk").fetchone()[0]
        )
        if eligible_count:
            risk_path = risk_parts / f"part-{pair_bucket:03d}.parquet"
            _copy_table(connection, "eligible_pair_risk", risk_path)
            connection.execute(
                """
                CREATE OR REPLACE TEMP TABLE eligible_pair_history AS
                WITH daily AS (
                    SELECT e.dc_id_1, e.dc_id_2, e.report_date,
                           count(*) FILTER (WHERE e.is_serious = 1)::BIGINT AS a_on_date,
                           count(*) FILTER (WHERE e.is_serious = 0)::BIGINT AS b_on_date
                    FROM pair_exposures e
                    JOIN eligible_pair_risk r USING (dc_id_1, dc_id_2)
                    WHERE e.is_fit_train = 1 AND e.report_date IS NOT NULL
                    GROUP BY e.dc_id_1, e.dc_id_2, e.report_date
                )
                SELECT dc_id_1, dc_id_2, report_date,
                       sum(a_on_date) OVER (
                           PARTITION BY dc_id_1, dc_id_2 ORDER BY report_date
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                       )::BIGINT AS a,
                       sum(b_on_date) OVER (
                           PARTITION BY dc_id_1, dc_id_2 ORDER BY report_date
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                       )::BIGINT AS b
                FROM daily
                """
            )
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE pair_scores AS
                WITH fit_exposures AS (
                    SELECT e.primaryid, e.dc_id_1, e.dc_id_2, e.report_date
                    FROM pair_exposures e
                    JOIN eligible_pair_risk r USING (dc_id_1, dc_id_2)
                    WHERE e.is_fit_train = 1 AND e.report_date IS NOT NULL
                ), with_totals AS (
                    SELECT e.primaryid, e.dc_id_1, e.dc_id_2, e.report_date,
                           t.total_serious, t.total_nonserious
                    FROM fit_exposures e
                    ASOF LEFT JOIN training_total_history t
                      ON e.report_date > t.report_date
                ), temporal AS (
                    SELECT e.primaryid, e.dc_id_1, e.dc_id_2,
                           CASE
                               WHEN h.dc_id_1 IS NULL
                                    OR h.a + h.b < {int(minimum_pair_reports)}
                                    OR coalesce(e.total_serious, 0) = 0
                                    OR coalesce(e.total_nonserious, 0) = 0
                               THEN NULL
                               ELSE ln(
                                   ((h.a + 0.5)::DOUBLE *
                                    (e.total_nonserious - h.b + 0.5)::DOUBLE) /
                                   ((h.b + 0.5)::DOUBLE *
                                    (e.total_serious - h.a + 0.5)::DOUBLE)
                               )::DOUBLE
                           END AS log_ror
                    FROM with_totals e
                    ASOF LEFT JOIN eligible_pair_history h
                      ON e.dc_id_1 = h.dc_id_1
                     AND e.dc_id_2 = h.dc_id_2
                     AND e.report_date > h.report_date
                )
                SELECT primaryid, dc_id_1, dc_id_2, log_ror
                FROM temporal WHERE log_ror IS NOT NULL
                UNION ALL
                SELECT e.primaryid, e.dc_id_1, e.dc_id_2, r.log_ror
                FROM pair_exposures e
                JOIN eligible_pair_risk r USING (dc_id_1, dc_id_2)
                WHERE e.is_fit_train = 0
                """
            )
            score_count = int(connection.execute("SELECT count(*) FROM pair_scores").fetchone()[0])
            if score_count:
                partial_destination = partials / f"pair_bucket={pair_bucket}"
                partial_destination.mkdir(parents=True, exist_ok=True)
                connection.execute(
                    f"""
                    COPY (
                        SELECT primaryid,
                               count(*) FILTER (
                                   WHERE log_ror > ln({PAIR_HIGH_RISK_ROR})
                               )::BIGINT AS high_risk_count_partial,
                               max(log_ror)::DOUBLE AS max_log_ror_partial,
                               sum(log_ror)::DOUBLE AS log_ror_sum_partial,
                               count(*)::BIGINT AS scored_count_partial,
                               cast(hash(primaryid) % {buckets} AS INTEGER)
                                   AS {_PATIENT_BUCKET}
                        FROM pair_scores GROUP BY primaryid
                    ) TO '{_sql_literal(partial_destination.as_posix())}' (
                        FORMAT PARQUET,
                        COMPRESSION SNAPPY,
                        PARTITION_BY ({_PATIENT_BUCKET}),
                        ROW_GROUP_SIZE 100000
                    )
                    """
                )
        _drop_pair_bucket_tables(connection)
        if completed == 1 or completed % 8 == 0 or completed == len(pair_buckets):
            print(
                f"  pair risk: pair bucket {completed}/{len(pair_buckets)} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )
    shutil.rmtree(pair_source, ignore_errors=True)


def _drop_pair_bucket_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table in (
        "pair_scores",
        "eligible_pair_history",
        "eligible_pair_risk",
        "pair_exposures",
    ):
        connection.execute(f"DROP TABLE IF EXISTS {table}")


def _create_top_pair_lookup(
    connection: duckdb.DuckDBPyConnection,
    *,
    work_dir: Path,
    top_pairs: int,
) -> None:
    risk_files = sorted((work_dir / "pair-risk-parts").glob("*.parquet"))
    if not risk_files:
        connection.execute(
            """
            CREATE TABLE top_pair_lookup (
                dc_id_1 BIGINT, dc_id_2 BIGINT, a BIGINT, b BIGINT,
                c BIGINT, d BIGINT, pair_reports BIGINT, prr DOUBLE,
                log_ror DOUBLE, risk_rank INTEGER
            )
            """
        )
        return
    connection.execute(
        f"""
        CREATE TABLE top_pair_lookup AS
        WITH bounded_top AS (
            SELECT dc_id_1, dc_id_2, a, b, c, d, pair_reports, prr, log_ror
            FROM {_read_parquet_sql(risk_files)}
            WHERE prr IS NOT NULL AND isfinite(prr)
            ORDER BY prr DESC, pair_reports DESC, dc_id_1, dc_id_2
            LIMIT {int(top_pairs)}
        )
        SELECT dc_id_1, dc_id_2, a, b, c, d, pair_reports, prr, log_ror,
               row_number() OVER (
                   ORDER BY prr DESC, pair_reports DESC, dc_id_1, dc_id_2
               )::INTEGER AS risk_rank
        FROM bounded_top
        ORDER BY risk_rank
        """
    )
    shutil.rmtree(work_dir / "pair-risk-parts", ignore_errors=True)


def _reduce_pair_partials(
    connection: duckdb.DuckDBPyConnection,
    *,
    work_dir: Path,
    buckets: int,
) -> None:
    partial_root = work_dir / "pair-patient-partials"
    output_root = work_dir / "patient-pair-features"
    output_root.mkdir(parents=True, exist_ok=True)
    patient_buckets = _nested_partition_values(partial_root, _PATIENT_BUCKET)
    for completed, bucket in enumerate(patient_buckets, start=1):
        files = sorted(
            path
            for path in partial_root.glob(
                f"pair_bucket=*/{_PATIENT_BUCKET}={bucket}/*.parquet"
            )
            if path.is_file()
        )
        if not files:
            continue
        destination = output_root / f"part-{bucket:03d}.parquet"
        connection.execute(
            f"""
            COPY (
                SELECT primaryid,
                       sum(high_risk_count_partial)::INTEGER AS num_high_risk_pairs,
                       max(max_log_ror_partial)::DOUBLE AS max_pair_log_ror,
                       (
                           sum(log_ror_sum_partial) /
                           nullif(sum(scored_count_partial), 0)
                       )::DOUBLE AS mean_pair_log_ror,
                       sum(scored_count_partial)::INTEGER AS scored_pair_count
                FROM {_read_parquet_sql(files)}
                GROUP BY primaryid
            ) TO '{_sql_literal(destination.as_posix())}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        _delete_files(files)
        if completed == 1 or completed % 8 == 0 or completed == len(patient_buckets):
            print(
                f"  pair reduce: patient bucket {completed}/{len(patient_buckets)}",
                flush=True,
            )
    shutil.rmtree(partial_root, ignore_errors=True)


def _build_indication_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    indi_files: list[Path],
    work_dir: Path,
    training_end_year: int,
    buckets: int,
) -> None:
    source_root = work_dir / "indication-source"
    vocabulary_root = work_dir / "indication-vocabulary-parts"
    feature_root = work_dir / "patient-indication-features"
    source_root.mkdir(parents=True, exist_ok=True)
    vocabulary_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
            SELECT primaryid, upper(trim(indi_pt)) AS indication,
                   cast(hash(primaryid) % {buckets} AS INTEGER) AS {_PATIENT_BUCKET}
            FROM {_read_parquet_sql(indi_files)}
            WHERE indi_pt IS NOT NULL AND trim(indi_pt) <> '' AND primaryid IS NOT NULL
        ) TO '{_sql_literal(source_root.as_posix())}' (
            FORMAT PARQUET,
            COMPRESSION SNAPPY,
            PARTITION_BY ({_PATIENT_BUCKET}),
            ROW_GROUP_SIZE 100000
        )
        """
    )
    patient_buckets = _partition_values(work_dir / "base-buckets", _PATIENT_BUCKET)
    for bucket in patient_buckets:
        source_files = _partition_files(source_root, _PATIENT_BUCKET, bucket)
        base_files = _partition_files(work_dir / "base-buckets", _PATIENT_BUCKET, bucket)
        if not source_files or not base_files:
            continue
        destination = vocabulary_root / f"part-{bucket:03d}.parquet"
        connection.execute(
            f"""
            COPY (
                SELECT DISTINCT i.indication
                FROM {_read_parquet_sql(source_files)} i
                JOIN {_read_parquet_sql(base_files)} c USING (primaryid)
                WHERE c.split = 'train'
                  AND year(c.report_date) <= {int(training_end_year)}
            ) TO '{_sql_literal(destination.as_posix())}' (
                FORMAT PARQUET, COMPRESSION SNAPPY
            )
            """
        )

    vocabulary_files = sorted(vocabulary_root.glob("*.parquet"))
    if vocabulary_files:
        connection.execute(
            f"""
            CREATE TABLE indication_lookup AS
            SELECT DISTINCT indication,
                   mod(hash(indication), {INDICATION_HASH_BINS})::INTEGER AS hash_bin,
                   regexp_matches(indication, '{MALIGNANCY_PATTERN}')::INTEGER
                       AS has_malignancy,
                   regexp_matches(indication, '{CARDIO_PATTERN}')::INTEGER AS has_cardio,
                   regexp_matches(indication, '{INFECTION_PATTERN}')::INTEGER
                       AS has_infection
            FROM {_read_parquet_sql(vocabulary_files)}
            """
        )
    else:
        connection.execute(
            """
            CREATE TABLE indication_lookup (
                indication VARCHAR, hash_bin INTEGER, has_malignancy INTEGER,
                has_cardio INTEGER, has_infection INTEGER
            )
            """
        )
    shutil.rmtree(vocabulary_root, ignore_errors=True)

    hash_counts = ",\n".join(
        f"count(*) FILTER (WHERE hash_bin = {index})::INTEGER "
        f"AS indication_hash_{index:02d}"
        for index in range(INDICATION_HASH_BINS)
    )
    for completed, bucket in enumerate(patient_buckets, start=1):
        source_files = _partition_files(source_root, _PATIENT_BUCKET, bucket)
        base_files = _partition_files(work_dir / "base-buckets", _PATIENT_BUCKET, bucket)
        if not source_files or not base_files:
            continue
        destination = feature_root / f"part-{bucket:03d}.parquet"
        connection.execute(
            f"""
            COPY (
                WITH deduplicated AS (
                    SELECT DISTINCT i.primaryid, i.indication, l.hash_bin,
                           l.has_malignancy, l.has_cardio, l.has_infection
                    FROM {_read_parquet_sql(source_files)} i
                    JOIN indication_lookup l USING (indication)
                    JOIN {_read_parquet_sql(base_files)} c USING (primaryid)
                )
                SELECT primaryid,
                       max(has_malignancy)::INTEGER AS has_malignancy,
                       max(has_cardio)::INTEGER AS has_cardio,
                       max(has_infection)::INTEGER AS has_infection,
                       {hash_counts}
                FROM deduplicated GROUP BY primaryid
            ) TO '{_sql_literal(destination.as_posix())}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        _delete_files(source_files)
        if completed == 1 or completed % 8 == 0 or completed == len(patient_buckets):
            print(
                f"  indications: patient bucket {completed}/{len(patient_buckets)}",
                flush=True,
            )
    shutil.rmtree(source_root, ignore_errors=True)


def _write_bucketed_enriched_cohort(
    connection: duckdb.DuckDBPyConnection,
    *,
    work_dir: Path,
    destination: Path,
) -> None:
    base_root = work_dir / "base-buckets"
    pair_root = work_dir / "patient-pair-features"
    indication_root = work_dir / "patient-indication-features"
    output_root = work_dir / "enriched-parts"
    output_root.mkdir(parents=True, exist_ok=True)
    patient_buckets = _partition_values(base_root, _PATIENT_BUCKET)
    indication_hash_select = "".join(
        f", coalesce(i.indication_hash_{index:02d}, 0)::INTEGER "
        f"AS indication_hash_{index:02d}"
        for index in range(INDICATION_HASH_BINS)
    )
    for completed, bucket in enumerate(patient_buckets, start=1):
        base_files = _partition_files(base_root, _PATIENT_BUCKET, bucket)
        base_relation = _read_parquet_sql(base_files)
        pair_path = pair_root / f"part-{bucket:03d}.parquet"
        indication_path = indication_root / f"part-{bucket:03d}.parquet"
        pair_relation = _patient_pair_relation(pair_path, base_relation=base_relation)
        indication_relation = _patient_indication_relation(
            indication_path,
            base_relation=base_relation,
        )
        output_path = output_root / f"part-{bucket:03d}.parquet"
        connection.execute(
            f"""
            COPY (
                SELECT c.*,
                       coalesce(i.has_malignancy, 0)::INTEGER AS has_malignancy,
                       coalesce(i.has_cardio, 0)::INTEGER AS has_cardio,
                       coalesce(i.has_infection, 0)::INTEGER AS has_infection,
                       coalesce(p.num_high_risk_pairs, 0)::INTEGER AS num_high_risk_pairs,
                       coalesce(p.max_pair_log_ror, 0.0)::DOUBLE AS max_pair_log_ror,
                       coalesce(p.mean_pair_log_ror, 0.0)::DOUBLE AS mean_pair_log_ror,
                       coalesce(p.scored_pair_count, 0)::INTEGER AS scored_pair_count
                       {indication_hash_select},
                       list_contains(
                           string_split(coalesce(c.outcome_codes, ''), '|'), 'DE'
                       )::INTEGER AS is_death,
                       list_contains(
                           string_split(coalesce(c.outcome_codes, ''), '|'), 'HO'
                       )::INTEGER AS is_hospitalization
                FROM {base_relation} c
                LEFT JOIN {indication_relation} i USING (primaryid)
                LEFT JOIN {pair_relation} p USING (primaryid)
            ) TO '{_sql_literal(output_path.as_posix())}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        _delete_files(base_files)
        pair_path.unlink(missing_ok=True)
        indication_path.unlink(missing_ok=True)
        if completed == 1 or completed % 8 == 0 or completed == len(patient_buckets):
            print(
                f"  final join: patient bucket {completed}/{len(patient_buckets)}",
                flush=True,
            )
    output_files = sorted(output_root.glob("*.parquet"))
    if not output_files:
        raise FeatureRescueError("feature-rescue patient partitioning produced no output")
    connection.execute(
        f"""
        COPY (
            SELECT * FROM {_read_parquet_sql(output_files)}
        ) TO '{_sql_literal(destination.as_posix())}' (
            FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
        )
        """
    )


def _patient_pair_relation(path: Path, *, base_relation: str) -> str:
    if path.is_file():
        return _read_parquet_sql([path])
    return f"""(
        SELECT primaryid,
               0::INTEGER AS num_high_risk_pairs,
               0.0::DOUBLE AS max_pair_log_ror,
               0.0::DOUBLE AS mean_pair_log_ror,
               0::INTEGER AS scored_pair_count
        FROM {base_relation} WHERE false
    )"""


def _patient_indication_relation(path: Path, *, base_relation: str) -> str:
    if path.is_file():
        return _read_parquet_sql([path])
    hash_columns = ", ".join(
        f"0::INTEGER AS indication_hash_{index:02d}" for index in range(INDICATION_HASH_BINS)
    )
    return f"""(
        SELECT primaryid,
               0::INTEGER AS has_malignancy,
               0::INTEGER AS has_cardio,
               0::INTEGER AS has_infection,
               {hash_columns}
        FROM {base_relation} WHERE false
    )"""


def _copy_table(connection: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY (SELECT * FROM {table}) TO '{_sql_literal(path.as_posix())}' "
        "(FORMAT PARQUET, COMPRESSION SNAPPY)"
    )


def _read_parquet_sql(files: list[Path]) -> str:
    if not files:
        raise FeatureRescueError("cannot read an empty Parquet shard list")
    paths = ", ".join(f"'{_sql_literal(path.as_posix())}'" for path in files)
    return (
        f"read_parquet([{paths}], union_by_name = true, "
        "hive_partitioning = false)"
    )


def _partition_values(root: Path, column: str) -> list[int]:
    values: list[int] = []
    for path in root.glob(f"{column}=*"):
        if not path.is_dir():
            continue
        try:
            values.append(int(path.name.split("=", 1)[1]))
        except ValueError:
            continue
    return sorted(set(values))


def _nested_partition_values(root: Path, column: str) -> list[int]:
    values: list[int] = []
    for path in root.glob(f"*/{column}=*"):
        if not path.is_dir():
            continue
        try:
            values.append(int(path.name.split("=", 1)[1]))
        except ValueError:
            continue
    return sorted(set(values))


def _partition_files(root: Path, column: str, value: int) -> list[Path]:
    return sorted(
        path for path in (root / f"{column}={value}").glob("*.parquet") if path.is_file()
    )


def _delete_files(files: list[Path]) -> None:
    parents: set[Path] = set()
    for path in files:
        parents.add(path.parent)
        path.unlink(missing_ok=True)
    for parent in sorted(parents, key=lambda item: len(item.parts), reverse=True):
        with suppress(OSError):
            parent.rmdir()


def _verify_output(path: Path, *, expected_rows: int) -> None:
    parquet = pq.ParquetFile(path)
    required = {
        "has_malignancy",
        "has_cardio",
        "has_infection",
        "num_high_risk_pairs",
        "max_pair_log_ror",
        "mean_pair_log_ror",
        "scored_pair_count",
        "is_death",
        "is_hospitalization",
        *(f"indication_hash_{index:02d}" for index in range(INDICATION_HASH_BINS)),
    }
    forbidden = {_PATIENT_BUCKET, _PAIR_BUCKET}
    columns = set(parquet.schema_arrow.names)
    if (
        parquet.metadata.num_rows != expected_rows
        or not required.issubset(columns)
        or forbidden & columns
    ):
        raise FeatureRescueError("invalid feature-rescue cohort schema or row count")
    _verify_snappy(path, label="feature-rescue cohort")


def _verify_snappy(path: Path, *, label: str) -> None:
    parquet = pq.ParquetFile(path)
    for row_group in range(parquet.metadata.num_row_groups):
        for column in range(parquet.metadata.num_columns):
            if parquet.metadata.row_group(row_group).column(column).compression != "SNAPPY":
                raise FeatureRescueError(f"{label} contains a non-Snappy column")


def _write_manifest(
    path: Path,
    *,
    record: FeatureRescueRecord,
    input_checksums: dict[str, str],
    training_end_year: int,
    minimum_pair_reports: int,
) -> None:
    payload = {
        "dataset": "TekaRx feature rescue",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "input_sha256": input_checksums,
        "training_end_year": training_end_year,
        "minimum_pair_reports": minimum_pair_reports,
        "temporal_policy": PAIR_RISK_SCOPE,
        "pair_encoding": {
            "score": "Haldane-Anscombe-smoothed log ROR",
            "high_risk_threshold_ror": PAIR_HIGH_RISK_ROR,
            "same_date_policy": "excluded from training encodings",
            "materialization": (
                "eligible pair/date histories sharded by DrugCentral pair; "
                "patient aggregates reduced from bounded partials"
            ),
        },
        "indication_encoding": {
            "vocabulary": "exact terms observed in fit-train only",
            "hash_bins": INDICATION_HASH_BINS,
            "held_out_unknown_terms": "ignored",
            "deduplication": "one (primaryid, indication) row before flags and hash counts",
        },
        "memory_strategy": {
            "patient_hash_buckets": FEATURE_RESCUE_BUCKETS,
            "pair_hash_buckets": FEATURE_RESCUE_BUCKETS,
            "global_patient_pair_table": False,
            "global_final_sort": False,
        },
        "prospective_exclusions": [
            "reactions",
            "reporter type",
            "drug role",
            "dechallenge/rechallenge",
            "outcomes as predictors",
        ],
        "auxiliary_targets": ["is_death", "is_hospitalization"],
        "record": asdict(record),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")
