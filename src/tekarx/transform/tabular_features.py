"""Add leakage-safe patient-level medication features and rebuild the graph."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from tekarx.extract.common import sha256_file
from tekarx.transform.dosage import (
    DOSAGE_PATIENT_FEATURES,
    DOSE_NORMALIZATION_SCOPE,
    DRUG_DOSE_MIN_SUPPORT,
    RELATIVE_LOG_DOSE_CLIP,
    dose_dimension_case_sql,
    dose_factor_case_sql,
    frequency_case_sql,
    normalized_frequency_sql,
    normalized_unit_sql,
)
from tekarx.transform.duckdb_runtime import configure_duckdb
from tekarx.transform.graph import GraphBuildRecord, build_graph

ATC_INITIALS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TABULAR_FEATURE_BUCKETS = 64
DRUG_RISK_SCOPE = (
    "strict-prior-date temporal encoding for train; frozen full-train lookup for "
    "validation/test"
)


class TabularFeatureBuildError(RuntimeError):
    """Raised when enriched-cohort invariants fail."""


@dataclass(frozen=True)
class TabularFeatureBuildRecord:
    """Summary for one enriched cohort and optional graph rebuild."""

    output_path: str
    rows: int
    compression: str
    ror_scope: str
    drug_risk_lookup_path: str
    dosage_normalization_scope: str
    dosage_edge_path: str
    dosage_lookup_path: str
    graph_path: str | None
    validation_auc: float | None


def add_tabular_features(
    *,
    data_dir: Path,
    memory_limit: str = "4GB",
    threads: int | None = None,
    rebuild_graph: bool = True,
) -> TabularFeatureBuildRecord:
    """Create a prospective enriched cohort with cross-fitted target encodings."""
    cohort = data_dir / "processed" / "tekarx_cohort.parquet"
    dictionary = data_dir / "processed" / "drug_dictionary.parquet"
    report_drug = data_dir / "processed" / "edges" / "report_drug.parquet"
    for path in (cohort, dictionary, report_drug):
        if not path.is_file():
            raise TabularFeatureBuildError(f"missing required input: {path}")
    output = data_dir / "processed" / "tekarx_cohort_enriched.parquet"
    risk_lookup = data_dir / "processed" / "drug_risk_lookup.parquet"
    dosage_edge = data_dir / "processed" / "edges" / "report_drug_dose.parquet"
    dosage_lookup = data_dir / "processed" / "dose_normalization_lookup.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    risk_temporary = risk_lookup.with_suffix(".parquet.tmp")
    dosage_edge_temporary = dosage_edge.with_suffix(".parquet.tmp")
    dosage_lookup_temporary = dosage_lookup.with_suffix(".parquet.tmp")
    database = data_dir / "interim" / f".tabular-features-{os.getpid()}.duckdb"
    database.parent.mkdir(parents=True, exist_ok=True)
    dosage_edge.parent.mkdir(parents=True, exist_ok=True)
    for path in (
        temporary,
        risk_temporary,
        dosage_edge_temporary,
        dosage_lookup_temporary,
    ):
        path.unlink(missing_ok=True)
    database.unlink(missing_ok=True)
    work_dir: Path | None = None
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(database))
        configure_duckdb(
            connection,
            data_dir=data_dir,
            stage="tabular-features",
            memory_limit=memory_limit,
            threads=threads,
        )
        work_dir = Path(
            tempfile.mkdtemp(prefix=".tabular-features-", dir=data_dir / "interim")
        )
        _validate_input_columns(cohort=cohort, report_drug=report_drug)
        _build_feature_tables(
            connection,
            cohort=cohort,
            dictionary=dictionary,
            report_drug=report_drug,
            work_dir=work_dir,
            buckets=TABULAR_FEATURE_BUCKETS,
        )
        print("Tabular [5/7]: writing frozen risk and dosage artifacts", flush=True)
        connection.execute(
            f"COPY (SELECT * FROM full_train_drug_risk ORDER BY dc_id) TO "
            f"'{_sql_literal(risk_temporary.as_posix())}' "
            "(FORMAT PARQUET, COMPRESSION SNAPPY)"
        )
        connection.execute(
            f"COPY (SELECT * FROM dose_normalization_lookup "
            "ORDER BY reference_level, dose_dimension, dc_id) TO "
            f"'{_sql_literal(dosage_lookup_temporary.as_posix())}' "
            "(FORMAT PARQUET, COMPRESSION SNAPPY)"
        )
        connection.execute(
            f"COPY (SELECT * FROM normalized_dose_edges) TO "
            f"'{_sql_literal(dosage_edge_temporary.as_posix())}' "
            "(FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000)"
        )
        print("Tabular [6/7]: assembling the enriched patient cohort", flush=True)
        _write_bucketed_enriched_cohort(
            connection,
            work_dir=work_dir,
            buckets=TABULAR_FEATURE_BUCKETS,
            destination=temporary,
        )
        expected_rows = connection.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_literal(cohort.as_posix())}')"
        ).fetchone()[0]
        expected_edge_rows = connection.execute(
            f"SELECT count(*) FROM read_parquet("
            f"'{_sql_literal(report_drug.as_posix())}')"
        ).fetchone()[0]
    except BaseException:
        for path in (
            temporary,
            risk_temporary,
            dosage_edge_temporary,
            dosage_lookup_temporary,
        ):
            path.unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:
            connection.close()
        database.unlink(missing_ok=True)
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    print("Tabular [7/7]: validating atomic Snappy Parquet outputs", flush=True)
    _verify_enriched(temporary, expected_rows=expected_rows)
    _verify_risk_lookup(risk_temporary)
    _verify_dose_edges(dosage_edge_temporary, expected_rows=expected_edge_rows)
    _verify_dose_lookup(dosage_lookup_temporary)
    os.replace(temporary, output)
    os.replace(risk_temporary, risk_lookup)
    os.replace(dosage_edge_temporary, dosage_edge)
    os.replace(dosage_lookup_temporary, dosage_lookup)
    graph_record: GraphBuildRecord | None = None
    if rebuild_graph:
        graph_record = build_graph(
            data_dir=data_dir,
            memory_limit=memory_limit,
            threads=threads,
            cohort_path=output,
        )
    record = TabularFeatureBuildRecord(
        output_path=str(output),
        rows=expected_rows,
        compression="snappy",
        ror_scope=DRUG_RISK_SCOPE,
        drug_risk_lookup_path=str(risk_lookup),
        dosage_normalization_scope=DOSE_NORMALIZATION_SCOPE,
        dosage_edge_path=str(dosage_edge),
        dosage_lookup_path=str(dosage_lookup),
        graph_path=graph_record.graph_path if graph_record else None,
        validation_auc=graph_record.validation_auc if graph_record else None,
    )
    _write_manifest(
        output.with_name("cohort_enriched_manifest.json"),
        record=record,
        inputs={
            str(cohort): sha256_file(cohort),
            str(dictionary): sha256_file(dictionary),
            str(report_drug): sha256_file(report_drug),
        },
        outputs={
            str(output): sha256_file(output),
            str(risk_lookup): sha256_file(risk_lookup),
            str(dosage_edge): sha256_file(dosage_edge),
            str(dosage_lookup): sha256_file(dosage_lookup),
        },
    )
    return record


def _build_feature_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    cohort: Path,
    dictionary: Path,
    report_drug: Path,
    work_dir: Path,
    buckets: int,
) -> None:
    cohort_sql = _sql_literal(cohort.as_posix())
    dictionary_sql = _sql_literal(dictionary.as_posix())
    report_drug_sql = _sql_literal(report_drug.as_posix())
    weight_kg = """
        CASE
            WHEN raw_weight IS NULL OR raw_weight <= 0 THEN NULL
            WHEN weight_unit IN ('KG','KGS','KILOGRAM','KILOGRAMS') THEN raw_weight
            WHEN weight_unit IN ('LB','LBS','POUND','POUNDS') THEN raw_weight * 0.45359237
            WHEN weight_unit IN ('G','GM','GMS','GRAM','GRAMS') THEN raw_weight / 1000.0
            WHEN weight_unit IN ('OZ','OUNCE','OUNCES') THEN raw_weight * 0.028349523125
            ELSE NULL
        END
    """
    print("Tabular [0/7]: building train-fitted demographic features", flush=True)
    connection.execute(
        f"""
        CREATE TABLE patient_demographic_raw AS
        WITH converted AS (
            SELECT primaryid,
                   CASE WHEN try_cast(age_years AS DOUBLE) BETWEEN 0 AND 120
                        THEN try_cast(age_years AS DOUBLE) END AS valid_age_years,
                   upper(trim(coalesce(sex, ''))) AS normalized_sex,
                   try_cast(weight AS DOUBLE) AS raw_weight,
                   upper(trim(coalesce(weight_unit, ''))) AS weight_unit
            FROM read_parquet('{cohort_sql}')
        ), normalized AS (
            SELECT primaryid, valid_age_years, normalized_sex, {weight_kg} AS weight_kg
            FROM converted
        )
        SELECT primaryid, valid_age_years, normalized_sex,
               CASE WHEN weight_kg BETWEEN 0.2 AND 500 THEN weight_kg END AS weight_kg
        FROM normalized
        """
    )
    connection.execute(
        f"""
        CREATE TABLE training_medians AS
        SELECT coalesce(median(d.valid_age_years), 50.0)::DOUBLE AS median_age_years,
               coalesce(median(d.weight_kg), 70.0)::DOUBLE AS median_weight_kg
        FROM patient_demographic_raw d
        JOIN read_parquet('{cohort_sql}') c USING (primaryid)
        WHERE c.split = 'train'
        """
    )
    connection.execute(
        """
        CREATE TABLE patient_demographic_features AS
        SELECT d.primaryid,
               coalesce(d.valid_age_years, m.median_age_years)::DOUBLE
                   AS age_imputed_years,
               (d.valid_age_years IS NULL)::INTEGER AS age_missing,
               (d.normalized_sex NOT IN ('M', 'F'))::INTEGER AS sex_unknown,
               least(greatest(coalesce(d.weight_kg, m.median_weight_kg) / 250.0, 0.0), 1.0)
                   ::DOUBLE AS weight_kg_normalized,
               (d.weight_kg IS NULL)::INTEGER AS weight_missing
        FROM patient_demographic_raw d CROSS JOIN training_medians m
        """
    )
    cohort_buckets = _partition_cohort(
        connection,
        cohort_sql=cohort_sql,
        cohort_passthrough=_cohort_passthrough_sql(cohort),
        work_dir=work_dir,
        buckets=buckets,
    )
    _create_bucketed_patient_drugs(
        connection,
        dictionary_sql=dictionary_sql,
        report_drug_sql=report_drug_sql,
        work_dir=work_dir,
        buckets=buckets,
    )
    print("Tabular [2/7]: calculating leakage-safe drug-risk histories", flush=True)
    connection.execute(
        f"""
        CREATE TABLE train_totals_daily AS
        SELECT report_date,
               count(*) FILTER (WHERE is_serious = 1)::BIGINT AS serious_on_date,
               count(*) FILTER (WHERE is_serious = 0)::BIGINT AS nonserious_on_date
        FROM read_parquet('{cohort_sql}')
        WHERE split = 'train' AND report_date IS NOT NULL
        GROUP BY report_date
        """
    )
    connection.execute(
        """
        CREATE TABLE train_total_history AS
        SELECT report_date,
               sum(serious_on_date) OVER (
                   ORDER BY report_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS total_serious,
               sum(nonserious_on_date) OVER (
                   ORDER BY report_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS total_nonserious
        FROM train_totals_daily
        """
    )
    connection.execute(
        f"""
        CREATE TABLE train_drug_daily AS
        SELECT p.dc_id, c.report_date,
               count(*) FILTER (WHERE c.is_serious = 1)::BIGINT AS a_on_date,
               count(*) FILTER (WHERE c.is_serious = 0)::BIGINT AS b_on_date
        FROM patient_drugs p
        JOIN read_parquet('{cohort_sql}') c USING (primaryid)
        WHERE c.split = 'train' AND c.report_date IS NOT NULL
        GROUP BY p.dc_id, c.report_date
        """
    )
    connection.execute(
        """
        CREATE TABLE train_drug_history AS
        SELECT dc_id, report_date,
               sum(a_on_date) OVER (
                   PARTITION BY dc_id ORDER BY report_date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS a,
               sum(b_on_date) OVER (
                   PARTITION BY dc_id ORDER BY report_date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS b
        FROM train_drug_daily
        """
    )
    connection.execute(
        f"""
        CREATE TABLE full_train_drug_risk AS
        WITH totals AS (
            SELECT count(*) FILTER (WHERE is_serious = 1) AS total_serious,
                   count(*) FILTER (WHERE is_serious = 0) AS total_nonserious
            FROM read_parquet('{cohort_sql}') WHERE split = 'train'
        ), counts AS (
            SELECT e.dc_id,
                   count(*) FILTER (WHERE c.is_serious = 1) AS a,
                   count(*) FILTER (WHERE c.is_serious = 0) AS b
            FROM patient_drugs e
            JOIN read_parquet('{cohort_sql}') c USING (primaryid)
            WHERE c.split = 'train'
            GROUP BY e.dc_id
        ), cells AS (
            SELECT dc_id, a, b, total_serious - a AS c, total_nonserious - b AS d,
                   total_serious, total_nonserious
            FROM counts CROSS JOIN totals
        )
        SELECT dc_id, a::BIGINT AS a, b::BIGINT AS b, c::BIGINT AS c, d::BIGINT AS d,
               (a + b)::BIGINT AS support,
               CASE WHEN total_serious = 0 OR total_nonserious = 0 THEN 1.0
                    ELSE ((a + 0.5)::DOUBLE * (d + 0.5)::DOUBLE) /
                         ((b + 0.5)::DOUBLE * (c + 0.5)::DOUBLE)
               END::DOUBLE AS ror
        FROM cells
        """
    )
    connection.execute(
        f"""
        CREATE TABLE drug_metadata AS
        SELECT dc_id, max(has_boxed_warning)::INTEGER AS has_boxed_warning,
               string_agg(DISTINCT atc_code, '|') FILTER (
                   WHERE atc_code IS NOT NULL AND trim(atc_code) <> ''
               ) AS atc_code
        FROM read_parquet('{dictionary_sql}')
        WHERE dc_id > 0
        GROUP BY dc_id
        """
    )
    print("Tabular [3/7]: aggregating patient drug and ATC features", flush=True)
    _create_bucketed_drug_features(
        connection,
        work_dir=work_dir,
        cohort_buckets=cohort_buckets,
        buckets=buckets,
    )
    _create_bucketed_exposure_features(
        connection,
        report_drug_sql=report_drug_sql,
        work_dir=work_dir,
        buckets=buckets,
    )
    print("Tabular [4/7]: normalizing dosage using train-only references", flush=True)
    _build_dosage_tables(
        connection,
        cohort=cohort,
        dictionary=dictionary,
        report_drug=report_drug,
        work_dir=work_dir,
        buckets=buckets,
    )


def _partition_cohort(
    connection: duckdb.DuckDBPyConnection,
    *,
    cohort_sql: str,
    cohort_passthrough: str,
    work_dir: Path,
    buckets: int,
) -> Path:
    destination = work_dir / "cohort-buckets"
    destination.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
            SELECT {cohort_passthrough},
                   g.age_imputed_years, g.age_missing, g.sex_unknown,
                   g.weight_kg_normalized, g.weight_missing,
                   cast(hash(c.primaryid) % {buckets} AS INTEGER) AS bucket
            FROM read_parquet('{cohort_sql}') c
            JOIN patient_demographic_features g USING (primaryid)
        ) TO '{_sql_literal(destination.as_posix())}' (
            FORMAT PARQUET,
            COMPRESSION SNAPPY,
            PARTITION_BY (bucket),
            ROW_GROUP_SIZE 100000
        )
        """
    )
    return destination


def _create_bucketed_patient_drugs(
    connection: duckdb.DuckDBPyConnection,
    *,
    dictionary_sql: str,
    report_drug_sql: str,
    work_dir: Path,
    buckets: int,
) -> None:
    """Deduplicate patient-ingredient exposures with bounded hash states."""
    if buckets < 1:
        raise ValueError("tabular feature buckets must be positive")
    source_dir = work_dir / "patient-drugs" / "source"
    output_dir = work_dir / "patient-drugs" / "deduplicated"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(
        f"Tabular [1/7]: mapping patient drugs into {buckets} disk-backed buckets",
        flush=True,
    )
    connection.execute(
        f"""
        COPY (
            SELECT e.primaryid, d.dc_id::BIGINT AS dc_id,
                   cast(hash(e.primaryid) % {buckets} AS INTEGER) AS bucket
            FROM read_parquet('{report_drug_sql}') e
            JOIN read_parquet('{dictionary_sql}') d ON d.faers_raw = trim(e.drugname)
            WHERE e.drugname IS NOT NULL AND trim(e.drugname) <> '' AND d.dc_id > 0
        ) TO '{_sql_literal(source_dir.as_posix())}' (
            FORMAT PARQUET,
            COMPRESSION SNAPPY,
            PARTITION_BY (bucket),
            ROW_GROUP_SIZE 100000
        )
        """
    )
    completed = 0
    for bucket in range(buckets):
        bucket_dir = source_dir / f"bucket={bucket}"
        if not bucket_dir.is_dir() or not any(bucket_dir.glob("*.parquet")):
            continue
        source_glob = _sql_literal((bucket_dir / "*.parquet").as_posix())
        destination = _sql_literal((output_dir / f"part-{bucket:03d}.parquet").as_posix())
        connection.execute(
            f"""
            COPY (
                SELECT DISTINCT primaryid, dc_id
                FROM read_parquet('{source_glob}')
            ) TO '{destination}' (
                FORMAT PARQUET,
                COMPRESSION SNAPPY,
                ROW_GROUP_SIZE 100000
            )
            """
        )
        completed += 1
        if completed == 1 or completed % 8 == 0:
            print(f"  patient drugs: {completed}/{buckets} buckets", flush=True)
    outputs = sorted(output_dir.glob("*.parquet"))
    if outputs:
        output_glob = _sql_literal((output_dir / "*.parquet").as_posix())
        connection.execute(
            f"CREATE VIEW patient_drugs AS SELECT * FROM read_parquet('{output_glob}')"
        )
    else:
        connection.execute("CREATE TABLE patient_drugs(primaryid VARCHAR, dc_id BIGINT)")
    shutil.rmtree(source_dir, ignore_errors=True)
    print(
        f"Tabular [1/7]: patient-drug mapping complete in "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )


def _create_bucketed_drug_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    work_dir: Path,
    cohort_buckets: Path,
    buckets: int,
) -> None:
    patient_drug_dir = work_dir / "patient-drugs" / "deduplicated"
    aggregate_dir = work_dir / "patient-drug-aggregates"
    class_dir = work_dir / "patient-class-aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    class_dir.mkdir(parents=True, exist_ok=True)
    l1_counts = ",\n".join(
        f"count(DISTINCT dc_id) FILTER (WHERE atc_l1 = '{letter}')::INTEGER "
        f"AS atc_l1_count_{letter.lower()}"
        for letter in ATC_INITIALS
    )
    completed = 0
    for bucket in range(buckets):
        patient_drugs = patient_drug_dir / f"part-{bucket:03d}.parquet"
        cohort_dir = cohort_buckets / f"bucket={bucket}"
        if not patient_drugs.is_file() or not cohort_dir.is_dir():
            continue
        cohort_glob = _sql_literal((cohort_dir / "*.parquet").as_posix())
        patient_drug_path = _sql_literal(patient_drugs.as_posix())
        connection.execute("DROP TABLE IF EXISTS bucket_patient_drug_nodes")
        connection.execute(
            f"""
            CREATE TEMP TABLE bucket_patient_drug_nodes AS
            WITH exposures AS (
                SELECT p.primaryid, p.dc_id, c.report_date, c.split
                FROM read_parquet('{patient_drug_path}') p
                JOIN read_parquet('{cohort_glob}') c USING (primaryid)
            ), with_totals AS (
                SELECT e.*, t.total_serious, t.total_nonserious
                FROM exposures e
                ASOF LEFT JOIN train_total_history t ON e.report_date > t.report_date
            ), with_history AS (
                SELECT e.*, h.dc_id AS history_dc_id, h.a, h.b
                FROM with_totals e
                ASOF LEFT JOIN train_drug_history h
                  ON e.dc_id = h.dc_id AND e.report_date > h.report_date
            )
            SELECT e.primaryid, e.dc_id,
                   CASE
                       WHEN e.split <> 'train' THEN coalesce(f.ror, 1.0)
                       WHEN e.history_dc_id IS NULL OR coalesce(e.total_serious, 0) = 0
                            OR coalesce(e.total_nonserious, 0) = 0 THEN 1.0
                       ELSE ((e.a + 0.5)::DOUBLE *
                             (e.total_nonserious - e.b + 0.5)::DOUBLE) /
                            ((e.b + 0.5)::DOUBLE *
                             (e.total_serious - e.a + 0.5)::DOUBLE)
                   END::DOUBLE AS ror,
                   m.has_boxed_warning, m.atc_code
            FROM with_history e
            LEFT JOIN full_train_drug_risk f USING (dc_id)
            LEFT JOIN drug_metadata m USING (dc_id)
            """
        )
        aggregate_path = _sql_literal(
            (aggregate_dir / f"part-{bucket:03d}.parquet").as_posix()
        )
        connection.execute(
            f"""
            COPY (
                SELECT primaryid,
                       coalesce(max(ror), 1.0)::DOUBLE AS max_ror,
                       coalesce(avg(ln(greatest(ror, 1e-12))), 0.0)::DOUBLE
                           AS mean_log_ror,
                       count(*) FILTER (WHERE ror > 2)::INTEGER AS high_ror_count,
                       max(coalesce(has_boxed_warning, 0))::INTEGER AS has_boxed_warning,
                       {bucket}::INTEGER AS bucket
                FROM bucket_patient_drug_nodes GROUP BY primaryid
            ) TO '{aggregate_path}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        class_path = _sql_literal((class_dir / f"part-{bucket:03d}.parquet").as_posix())
        connection.execute(
            f"""
            COPY (
                WITH patient_atc_codes AS (
                    SELECT DISTINCT p.primaryid, p.dc_id, upper(trim(code)) AS atc_code
                    FROM bucket_patient_drug_nodes p,
                         unnest(string_split(coalesce(p.atc_code, ''), '|')) codes(code)
                    WHERE regexp_matches(upper(trim(code)), '^[A-Z][0-9]{{2}}')
                ), hierarchy AS (
                    SELECT primaryid, dc_id, substr(atc_code, 1, 1) AS atc_l1,
                           substr(atc_code, 1, 3) AS atc_l2,
                           CASE WHEN length(atc_code) >= 4
                                THEN substr(atc_code, 1, 4) END AS atc_l3,
                           CASE WHEN length(atc_code) >= 5
                                THEN substr(atc_code, 1, 5) END AS atc_l4
                    FROM patient_atc_codes
                ), l1_frequency AS (
                    SELECT primaryid, atc_l1, count(DISTINCT dc_id)::INTEGER AS frequency
                    FROM hierarchy GROUP BY primaryid, atc_l1
                ), l2_frequency AS (
                    SELECT primaryid, atc_l2, count(DISTINCT dc_id)::INTEGER AS frequency
                    FROM hierarchy GROUP BY primaryid, atc_l2
                ), duplicate_summary AS (
                    SELECT primaryid,
                           sum(greatest(frequency - 1, 0))::INTEGER
                               AS therapeutic_duplicates
                    FROM l1_frequency GROUP BY primaryid
                ), duplicate_l2_summary AS (
                    SELECT primaryid,
                           sum(greatest(frequency - 1, 0))::INTEGER
                               AS therapeutic_duplicates_l2
                    FROM l2_frequency GROUP BY primaryid
                ), hierarchy_summary AS (
                    SELECT primaryid, count(DISTINCT atc_l1)::INTEGER AS atc_diversity,
                           count(DISTINCT atc_l2)::INTEGER AS atc_l2_diversity,
                           count(DISTINCT atc_l3)::INTEGER AS atc_l3_diversity,
                           count(DISTINCT atc_l4)::INTEGER AS atc_l4_diversity,
                           count(DISTINCT dc_id) FILTER (
                               WHERE atc_l1 IN ('N', 'B', 'C', 'M')
                           )::INTEGER AS num_high_risk_atc,
                           count(DISTINCT dc_id) FILTER (
                               WHERE atc_l3 IN ('N02A', 'B01A', 'M01A', 'A10A')
                           )::INTEGER AS num_high_risk_atc_groups,
                           {l1_counts}
                    FROM hierarchy GROUP BY primaryid
                )
                SELECT h.*, d.therapeutic_duplicates, l.therapeutic_duplicates_l2,
                       {bucket}::INTEGER AS bucket
                FROM hierarchy_summary h
                JOIN duplicate_summary d USING (primaryid)
                JOIN duplicate_l2_summary l USING (primaryid)
            ) TO '{class_path}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        connection.execute("DROP TABLE bucket_patient_drug_nodes")
        completed += 1
        if completed == 1 or completed % 8 == 0:
            print(f"  patient drug/ATC features: {completed}/{buckets} buckets", flush=True)

    _create_parquet_union_view(
        connection,
        table_name="patient_drug_aggregates",
        directory=aggregate_dir,
        empty_columns=(
            "primaryid VARCHAR, max_ror DOUBLE, mean_log_ror DOUBLE, "
            "high_ror_count INTEGER, has_boxed_warning INTEGER, bucket INTEGER"
        ),
    )
    class_columns = (
        "primaryid VARCHAR, atc_diversity INTEGER, atc_l2_diversity INTEGER, "
        "atc_l3_diversity INTEGER, atc_l4_diversity INTEGER, "
        "num_high_risk_atc INTEGER, num_high_risk_atc_groups INTEGER, "
        + ", ".join(f"atc_l1_count_{letter.lower()} INTEGER" for letter in ATC_INITIALS)
        + ", therapeutic_duplicates INTEGER, therapeutic_duplicates_l2 INTEGER, "
        "bucket INTEGER"
    )
    _create_parquet_union_view(
        connection,
        table_name="patient_class_aggregates",
        directory=class_dir,
        empty_columns=class_columns,
    )


def _create_parquet_union_view(
    connection: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    directory: Path,
    empty_columns: str,
) -> None:
    if any(directory.glob("*.parquet")):
        glob = _sql_literal((directory / "*.parquet").as_posix())
        connection.execute(
            f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{glob}')"
        )
    else:
        connection.execute(f"CREATE TABLE {table_name} ({empty_columns})")


def _write_bucketed_enriched_cohort(
    connection: duckdb.DuckDBPyConnection,
    *,
    work_dir: Path,
    buckets: int,
    destination: Path,
) -> None:
    """Join patient-sized feature tables one hash bucket at a time."""
    cohort_dir = work_dir / "cohort-buckets"
    output_dir = work_dir / "enriched-parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    atc_count_select = "".join(
        f", coalesce(k.atc_l1_count_{letter.lower()}, 0)::INTEGER "
        f"AS atc_l1_count_{letter.lower()}"
        for letter in ATC_INITIALS
    )
    completed = 0
    for bucket in range(buckets):
        cohort_bucket = cohort_dir / f"bucket={bucket}"
        if not cohort_bucket.is_dir() or not any(cohort_bucket.glob("*.parquet")):
            continue
        cohort_glob = _sql_literal((cohort_bucket / "*.parquet").as_posix())
        drug_source = _bucket_feature_source(
            work_dir / "patient-drug-aggregates",
            bucket=bucket,
            empty_table="patient_drug_aggregates",
        )
        class_source = _bucket_feature_source(
            work_dir / "patient-class-aggregates",
            bucket=bucket,
            empty_table="patient_class_aggregates",
        )
        exposure_source = _bucket_feature_source(
            work_dir / "exposures" / "aggregates",
            bucket=bucket,
            empty_table="patient_exposure_aggregates",
        )
        dosage_source = _bucket_feature_source(
            work_dir / "dosage-patient" / "aggregates",
            bucket=bucket,
            empty_table="patient_dosage_aggregates",
        )
        output_path = _sql_literal(
            (output_dir / f"part-{bucket:03d}.parquet").as_posix()
        )
        connection.execute(
            f"""
            COPY (
                SELECT c.*,
                       coalesce(a.max_ror, 1.0)::DOUBLE AS max_ror,
                       coalesce(a.mean_log_ror, 0.0)::DOUBLE AS mean_log_ror,
                       coalesce(a.high_ror_count, 0)::INTEGER AS high_ror_count,
                       coalesce(a.has_boxed_warning, 0)::INTEGER AS has_boxed_warning,
                       coalesce(k.num_high_risk_atc, 0)::INTEGER AS num_high_risk_atc,
                       coalesce(k.atc_diversity, 0)::INTEGER AS atc_diversity,
                       coalesce(k.therapeutic_duplicates, 0)::INTEGER
                           AS therapeutic_duplicates,
                       coalesce(k.atc_l2_diversity, 0)::INTEGER AS atc_l2_diversity,
                       coalesce(k.atc_l3_diversity, 0)::INTEGER AS atc_l3_diversity,
                       coalesce(k.atc_l4_diversity, 0)::INTEGER AS atc_l4_diversity,
                       coalesce(k.therapeutic_duplicates_l2, 0)::INTEGER
                           AS therapeutic_duplicates_l2,
                       coalesce(k.num_high_risk_atc_groups, 0)::INTEGER
                           AS num_high_risk_atc_groups
                       {atc_count_select},
                       coalesce(x.exposure_route_diversity, 0)::INTEGER
                           AS exposure_route_diversity,
                       coalesce(x.exposure_route_documented_fraction, 0.0)::DOUBLE
                           AS exposure_route_documented_fraction,
                       coalesce(x.exposure_has_oral_route, 0)::INTEGER
                           AS exposure_has_oral_route,
                       coalesce(x.exposure_has_parenteral_route, 0)::INTEGER
                           AS exposure_has_parenteral_route,
                       coalesce(x.exposure_has_topical_route, 0)::INTEGER
                           AS exposure_has_topical_route,
                       coalesce(x.exposure_dose_documented_fraction, 0.0)::DOUBLE
                           AS exposure_dose_documented_fraction,
                       coalesce(x.exposure_dose_unit_diversity, 0)::INTEGER
                           AS exposure_dose_unit_diversity,
                       coalesce(x.exposure_dose_form_diversity, 0)::INTEGER
                           AS exposure_dose_form_diversity,
                       coalesce(x.exposure_has_solid_form, 0)::INTEGER
                           AS exposure_has_solid_form,
                       coalesce(x.exposure_has_liquid_form, 0)::INTEGER
                           AS exposure_has_liquid_form,
                       coalesce(x.exposure_has_injectable_form, 0)::INTEGER
                           AS exposure_has_injectable_form,
                       coalesce(x.exposure_frequency_documented_fraction, 0.0)::DOUBLE
                           AS exposure_frequency_documented_fraction,
                       coalesce(x.exposure_frequency_diversity, 0)::INTEGER
                           AS exposure_frequency_diversity,
                       coalesce(z.dose_normalized_numeric_fraction, 0.0)::DOUBLE
                           AS dose_normalized_numeric_fraction,
                       coalesce(z.dose_normalized_scheduled_fraction, 0.0)::DOUBLE
                           AS dose_normalized_scheduled_fraction,
                       coalesce(z.dose_normalized_relative_available_fraction, 0.0)::DOUBLE
                           AS dose_normalized_relative_available_fraction,
                       coalesce(z.dose_normalized_relative_log_mean, 0.0)::DOUBLE
                           AS dose_normalized_relative_log_mean,
                       coalesce(z.dose_normalized_relative_log_max, 0.0)::DOUBLE
                           AS dose_normalized_relative_log_max,
                       coalesce(
                           z.dose_normalized_daily_relative_available_fraction, 0.0
                       )::DOUBLE AS dose_normalized_daily_relative_available_fraction,
                       coalesce(z.dose_normalized_daily_relative_log_mean, 0.0)::DOUBLE
                           AS dose_normalized_daily_relative_log_mean,
                       coalesce(z.dose_normalized_daily_relative_log_max, 0.0)::DOUBLE
                           AS dose_normalized_daily_relative_log_max,
                       coalesce(z.dose_normalized_above_train_p90_count, 0)::INTEGER
                           AS dose_normalized_above_train_p90_count,
                       coalesce(z.dose_normalized_above_train_p90_fraction, 0.0)::DOUBLE
                           AS dose_normalized_above_train_p90_fraction,
                       coalesce(z.dose_normalized_num_parenteral_drugs, 0)::INTEGER
                           AS dose_normalized_num_parenteral_drugs,
                       coalesce(z.dose_normalized_parenteral_fraction, 0.0)::DOUBLE
                           AS dose_normalized_parenteral_fraction,
                       coalesce(z.dose_normalized_has_iv, 0)::INTEGER
                           AS dose_normalized_has_iv,
                       coalesce(z.dose_normalized_has_sc, 0)::INTEGER
                           AS dose_normalized_has_sc,
                       coalesce(z.dose_normalized_has_im, 0)::INTEGER
                           AS dose_normalized_has_im,
                       (c.age_imputed_years < 18)::INTEGER AS age_group_0_17,
                       (c.age_imputed_years BETWEEN 18 AND 40)::INTEGER
                           AS age_group_18_40,
                       (c.age_imputed_years BETWEEN 41 AND 64)::INTEGER
                           AS age_group_41_64,
                       (c.age_imputed_years >= 65)::INTEGER AS age_group_65_plus,
                       (c.num_drugs * c.num_drugs)::INTEGER AS num_drugs_squared,
                       (c.num_drugs * c.age_imputed_years)::DOUBLE AS polypharmacy_age
                FROM read_parquet('{cohort_glob}', hive_partitioning = false) c
                LEFT JOIN {drug_source} a USING (primaryid)
                LEFT JOIN {class_source} k USING (primaryid)
                LEFT JOIN {exposure_source} x USING (primaryid)
                LEFT JOIN {dosage_source} z USING (primaryid)
            ) TO '{output_path}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        completed += 1
        if completed == 1 or completed % 8 == 0:
            print(f"  enriched cohort: {completed}/{buckets} buckets", flush=True)
    if not any(output_dir.glob("*.parquet")):
        raise TabularFeatureBuildError("cohort bucketing produced no output rows")
    output_glob = _sql_literal((output_dir / "*.parquet").as_posix())
    destination_sql = _sql_literal(destination.as_posix())
    connection.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet('{output_glob}', union_by_name = true)
        ) TO '{destination_sql}' (
            FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
        )
        """
    )


def _bucket_feature_source(
    directory: Path,
    *,
    bucket: int,
    empty_table: str,
) -> str:
    path = directory / f"part-{bucket:03d}.parquet"
    if path.is_file():
        literal = _sql_literal(path.as_posix())
        return f"(SELECT * EXCLUDE (bucket) FROM read_parquet('{literal}'))"
    return f"(SELECT * EXCLUDE (bucket) FROM {empty_table} WHERE false)"


def _create_bucketed_exposure_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    report_drug_sql: str,
    work_dir: Path,
    buckets: int,
) -> None:
    source_dir = work_dir / "exposures" / "source"
    output_dir = work_dir / "exposures" / "aggregates"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  exposure features: partitioning normalized edge rows", flush=True)
    connection.execute(
        f"""
        COPY (
            SELECT e.primaryid,
                   upper(trim(coalesce(e.route, ''))) AS route,
                   try_cast(e.dose_amt AS DOUBLE) AS dose_amount,
                   upper(trim(coalesce(e.dose_unit, ''))) AS dose_unit,
                   upper(trim(coalesce(e.dose_form, ''))) AS dose_form,
                   upper(trim(coalesce(e.dose_freq, ''))) AS dose_frequency,
                   cast(hash(e.primaryid) % {buckets} AS INTEGER) AS bucket
            FROM read_parquet('{report_drug_sql}') e
        ) TO '{_sql_literal(source_dir.as_posix())}' (
            FORMAT PARQUET, COMPRESSION SNAPPY, PARTITION_BY (bucket),
            ROW_GROUP_SIZE 100000
        )
        """
    )
    completed = 0
    for bucket in range(buckets):
        bucket_dir = source_dir / f"bucket={bucket}"
        if not bucket_dir.is_dir() or not any(bucket_dir.glob("*.parquet")):
            continue
        source_glob = _sql_literal((bucket_dir / "*.parquet").as_posix())
        destination = _sql_literal((output_dir / f"part-{bucket:03d}.parquet").as_posix())
        connection.execute(
            f"""
            COPY (
                WITH exposures AS (
                    SELECT DISTINCT * FROM read_parquet('{source_glob}')
                )
                SELECT primaryid,
                       count(DISTINCT nullif(route, ''))::INTEGER
                           AS exposure_route_diversity,
                       avg((route <> '')::INTEGER)::DOUBLE
                           AS exposure_route_documented_fraction,
                       max(regexp_matches(
                           route, 'ORAL|BY MOUTH|(^|[^A-Z])PO([^A-Z]|$)'
                       ))::INTEGER AS exposure_has_oral_route,
                       max(regexp_matches(
                           route,
                           'INTRAVENOUS|INTRAMUSCULAR|SUBCUTANEOUS|INFUSION|PARENTERAL|(^|[^A-Z])(IV|IM|SC)([^A-Z]|$)'
                       ))::INTEGER AS exposure_has_parenteral_route,
                       max(regexp_matches(
                           route, 'TOPICAL|DERMAL|CUTANEOUS|TRANSDERMAL'
                       ))::INTEGER AS exposure_has_topical_route,
                       avg((dose_amount IS NOT NULL OR dose_unit <> '')::INTEGER)::DOUBLE
                           AS exposure_dose_documented_fraction,
                       count(DISTINCT nullif(dose_unit, ''))::INTEGER
                           AS exposure_dose_unit_diversity,
                       count(DISTINCT nullif(dose_form, ''))::INTEGER
                           AS exposure_dose_form_diversity,
                       max(regexp_matches(dose_form, 'TABLET|CAPSULE|CAPLET'))::INTEGER
                           AS exposure_has_solid_form,
                       max(regexp_matches(
                           dose_form, 'SOLUTION|SUSPENSION|SYRUP|LIQUID'
                       ))::INTEGER AS exposure_has_liquid_form,
                       max(regexp_matches(
                           dose_form, 'INJECT|INFUS|PARENTERAL'
                       ))::INTEGER AS exposure_has_injectable_form,
                       avg((dose_frequency <> '')::INTEGER)::DOUBLE
                           AS exposure_frequency_documented_fraction,
                       count(DISTINCT nullif(dose_frequency, ''))::INTEGER
                           AS exposure_frequency_diversity,
                       {bucket}::INTEGER AS bucket
                FROM exposures GROUP BY primaryid
            ) TO '{destination}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        completed += 1
        if completed == 1 or completed % 8 == 0:
            print(f"  exposure features: {completed}/{buckets} buckets", flush=True)
    _create_parquet_union_view(
        connection,
        table_name="patient_exposure_aggregates",
        directory=output_dir,
        empty_columns=(
            "primaryid VARCHAR, exposure_route_diversity INTEGER, "
            "exposure_route_documented_fraction DOUBLE, exposure_has_oral_route INTEGER, "
            "exposure_has_parenteral_route INTEGER, exposure_has_topical_route INTEGER, "
            "exposure_dose_documented_fraction DOUBLE, "
            "exposure_dose_unit_diversity INTEGER, exposure_dose_form_diversity INTEGER, "
            "exposure_has_solid_form INTEGER, exposure_has_liquid_form INTEGER, "
            "exposure_has_injectable_form INTEGER, "
            "exposure_frequency_documented_fraction DOUBLE, "
            "exposure_frequency_diversity INTEGER, bucket INTEGER"
        ),
    )
    shutil.rmtree(source_dir, ignore_errors=True)


def _build_dosage_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    cohort: Path,
    dictionary: Path,
    report_drug: Path,
    work_dir: Path,
    buckets: int,
) -> None:
    """Build train-fitted dosage references, audited edges, and patient features."""
    cohort_sql = _sql_literal(cohort.as_posix())
    dictionary_sql = _sql_literal(dictionary.as_posix())
    report_drug_sql = _sql_literal(report_drug.as_posix())
    edge_columns = pq.ParquetFile(report_drug).schema_arrow.names
    derived_names = {
        "faers_raw",
        "dictionary_positive_dc_id_count",
        "dc_id",
        "ingredient_normalization_eligible",
        "dose_dimension",
        "canonical_unit",
        "canonical_amount",
        "administrations_per_day",
        "canonical_daily_amount",
        "amount_reference_level",
        "amount_reference_support",
        "relative_log_amount",
        "above_train_p90",
        "daily_reference_level",
        "daily_reference_support",
        "relative_log_daily_amount",
        "above_train_daily_p90",
    }
    collisions = sorted(set(edge_columns) & derived_names)
    if collisions:
        raise TabularFeatureBuildError(
            f"report-drug columns collide with dosage output fields: {collisions}"
        )
    passthrough = [
        f"e.{_sql_identifier(column)} AS {_sql_identifier(column)}"
        for column in edge_columns
        if column != "split"
    ]
    for optional in ("caseid", "drug_seq", "prod_ai"):
        if optional not in edge_columns:
            passthrough.append(f"NULL::VARCHAR AS {_sql_identifier(optional)}")
    passthrough.append("c.split AS split")
    edge_select = ",\n                   ".join(passthrough)
    unit_token = normalized_unit_sql("e.dose_unit")
    frequency_token = normalized_frequency_sql("e.dose_freq")
    dose_dimension = dose_dimension_case_sql("unit_token")
    dose_factor = dose_factor_case_sql("unit_token")
    frequency = frequency_case_sql("frequency_token")

    connection.execute(
        f"""
        CREATE TABLE dose_name_mapping AS
        SELECT trim(faers_raw) AS faers_raw,
               count(DISTINCT dc_id) FILTER (WHERE dc_id > 0)::INTEGER
                   AS dictionary_positive_dc_id_count,
               CASE WHEN count(DISTINCT dc_id) FILTER (WHERE dc_id > 0) = 1
                    THEN min(dc_id) FILTER (WHERE dc_id > 0) END::BIGINT AS dc_id
        FROM read_parquet('{dictionary_sql}')
        WHERE faers_raw IS NOT NULL AND trim(faers_raw) <> ''
        GROUP BY trim(faers_raw)
        """
    )
    dose_base_dir = work_dir / "dose-exposure-base"
    dose_base_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"  dosage: parsing exposure rows into {buckets} ingredient shards",
        flush=True,
    )
    connection.execute(
        f"""
        COPY (
            WITH edge_rows AS (
                SELECT {edge_select},
                       trim(e.drugname) AS faers_raw,
                       try_cast(e.dose_amt AS DOUBLE) AS numeric_amount,
                       {unit_token} AS unit_token,
                       {frequency_token} AS frequency_token
                FROM read_parquet('{report_drug_sql}') e
                JOIN read_parquet('{cohort_sql}') c USING (primaryid)
            ), parsed AS (
                SELECT e.*, {dose_dimension} AS dose_dimension,
                       {dose_factor} AS dose_factor,
                       ({frequency})::DOUBLE AS administrations_per_day
                FROM edge_rows e
            ), canonical AS (
                SELECT p.*,
                       CASE
                           WHEN numeric_amount > 0 AND isfinite(numeric_amount)
                                AND dose_factor IS NOT NULL
                                AND isfinite(numeric_amount * dose_factor)
                           THEN numeric_amount * dose_factor
                       END::DOUBLE AS canonical_amount
                FROM parsed p
            )
            SELECT c.* EXCLUDE (numeric_amount, unit_token, frequency_token, dose_factor),
                   coalesce(m.dictionary_positive_dc_id_count, 0)::INTEGER
                       AS dictionary_positive_dc_id_count,
                   m.dc_id,
                   (
                       m.dictionary_positive_dc_id_count = 1
                       AND m.dc_id IS NOT NULL
                       AND c.canonical_amount IS NOT NULL
                   )::INTEGER AS ingredient_normalization_eligible,
                   CASE c.dose_dimension
                       WHEN 'mass_mg' THEN 'mg'
                       WHEN 'mass_mg_per_kg' THEN 'mg/kg'
                       WHEN 'mass_mg_per_m2' THEN 'mg/m2'
                       WHEN 'volume_ml' THEN 'mL'
                       WHEN 'activity_iu' THEN 'IU'
                       WHEN 'activity_iu_per_kg' THEN 'IU/kg'
                       WHEN 'equivalent_meq' THEN 'mEq'
                       WHEN 'substance_mmol' THEN 'mmol'
                       WHEN 'radioactivity_mbq' THEN 'MBq'
                   END AS canonical_unit,
                   CASE WHEN c.canonical_amount IS NOT NULL
                              AND c.administrations_per_day IS NOT NULL
                              AND isfinite(c.canonical_amount * c.administrations_per_day)
                        THEN c.canonical_amount * c.administrations_per_day
                   END::DOUBLE AS canonical_daily_amount,
                   cast(
                       CASE WHEN m.dc_id IS NOT NULL THEN hash(m.dc_id)
                            ELSE hash(c.primaryid) END % {buckets}
                       AS INTEGER
                   ) AS dose_bucket
            FROM canonical c
            LEFT JOIN dose_name_mapping m USING (faers_raw)
        ) TO '{_sql_literal(dose_base_dir.as_posix())}' (
            FORMAT PARQUET, COMPRESSION SNAPPY, PARTITION_BY (dose_bucket),
            ROW_GROUP_SIZE 100000
        )
        """
    )
    dose_base_glob = _sql_literal((dose_base_dir / "*" / "*.parquet").as_posix())
    connection.execute(
        f"CREATE VIEW dose_exposure_base AS "
        f"SELECT * FROM read_parquet('{dose_base_glob}', hive_partitioning = true)"
    )
    stats_select = """
        count(canonical_amount)::BIGINT AS support,
        quantile_cont(
            ln(canonical_amount), [0.25, 0.50, 0.75, 0.90]
        ) AS amount_quantiles,
        count(canonical_daily_amount)::BIGINT AS daily_support,
        quantile_cont(
            ln(canonical_daily_amount), [0.25, 0.50, 0.75, 0.90]
        ) AS daily_quantiles
    """
    stats_projection = """
        support,
        amount_quantiles[2]::DOUBLE AS median_log_amount,
        amount_quantiles[1]::DOUBLE AS q25_log_amount,
        amount_quantiles[3]::DOUBLE AS q75_log_amount,
        (amount_quantiles[3] - amount_quantiles[1])::DOUBLE AS iqr_log_amount,
        amount_quantiles[4]::DOUBLE AS p90_log_amount,
        daily_support,
        daily_quantiles[2]::DOUBLE AS median_log_daily_amount,
        daily_quantiles[1]::DOUBLE AS q25_log_daily_amount,
        daily_quantiles[3]::DOUBLE AS q75_log_daily_amount,
        (daily_quantiles[3] - daily_quantiles[1])::DOUBLE AS iqr_log_daily_amount,
        daily_quantiles[4]::DOUBLE AS p90_log_daily_amount
    """
    print("  dosage: fitting bounded train-only quantiles by drug", flush=True)
    dose_stats_dir = work_dir / "dose-train-drug-stats"
    dose_stats_dir.mkdir(parents=True, exist_ok=True)
    stats_completed = 0
    for bucket in range(buckets):
        bucket_dir = dose_base_dir / f"dose_bucket={bucket}"
        if not bucket_dir.is_dir() or not any(bucket_dir.glob("*.parquet")):
            continue
        source_glob = _sql_literal((bucket_dir / "*.parquet").as_posix())
        stats_path = _sql_literal(
            (dose_stats_dir / f"part-{bucket:03d}.parquet").as_posix()
        )
        connection.execute(
            f"""
            COPY (
                WITH stats AS (
                    SELECT dc_id, dose_dimension, {stats_select}
                    FROM read_parquet('{source_glob}', hive_partitioning = false)
                    WHERE split = 'train' AND ingredient_normalization_eligible = 1
                    GROUP BY dc_id, dose_dimension
                )
                SELECT dc_id, dose_dimension, {stats_projection} FROM stats
            ) TO '{stats_path}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        stats_completed += 1
        if stats_completed == 1 or stats_completed % 8 == 0:
            print(f"    drug quantiles: {stats_completed}/{buckets} shards", flush=True)
    _create_parquet_union_view(
        connection,
        table_name="dose_train_drug_stats",
        directory=dose_stats_dir,
        empty_columns=(
            "dc_id BIGINT, dose_dimension VARCHAR, support BIGINT, "
            "median_log_amount DOUBLE, q25_log_amount DOUBLE, "
            "q75_log_amount DOUBLE, iqr_log_amount DOUBLE, p90_log_amount DOUBLE, "
            "daily_support BIGINT, median_log_daily_amount DOUBLE, "
            "q25_log_daily_amount DOUBLE, q75_log_daily_amount DOUBLE, "
            "iqr_log_daily_amount DOUBLE, p90_log_daily_amount DOUBLE"
        ),
    )
    print("  dosage: fitting bounded train-only fallback quantiles", flush=True)
    connection.execute(
        f"""
        CREATE TABLE dose_train_dimension_stats AS
        WITH stats AS (
            SELECT dose_dimension, {stats_select}
            FROM dose_exposure_base
            WHERE split = 'train' AND ingredient_normalization_eligible = 1
            GROUP BY dose_dimension
        )
        SELECT dose_dimension, {stats_projection} FROM stats
        """
    )
    print("  dosage: writing frozen normalization lookup", flush=True)
    connection.execute(
        """
        CREATE TABLE dose_normalization_lookup AS
        SELECT 'drug_with_dimension_fallback'::VARCHAR AS reference_level,
               d.dc_id, d.dose_dimension,
               d.support, d.median_log_amount, d.q25_log_amount, d.q75_log_amount,
               d.iqr_log_amount, d.p90_log_amount, d.daily_support,
               d.median_log_daily_amount, d.q25_log_daily_amount,
               d.q75_log_daily_amount, d.iqr_log_daily_amount,
               d.p90_log_daily_amount,
               g.support AS fallback_support,
               g.median_log_amount AS fallback_median_log_amount,
               g.q25_log_amount AS fallback_q25_log_amount,
               g.q75_log_amount AS fallback_q75_log_amount,
               g.iqr_log_amount AS fallback_iqr_log_amount,
               g.p90_log_amount AS fallback_p90_log_amount,
               g.daily_support AS fallback_daily_support,
               g.median_log_daily_amount AS fallback_median_log_daily_amount,
               g.q25_log_daily_amount AS fallback_q25_log_daily_amount,
               g.q75_log_daily_amount AS fallback_q75_log_daily_amount,
               g.iqr_log_daily_amount AS fallback_iqr_log_daily_amount,
               g.p90_log_daily_amount AS fallback_p90_log_daily_amount
        FROM dose_train_drug_stats d
        JOIN dose_train_dimension_stats g USING (dose_dimension)
        """
    )
    print("  dosage: applying frozen dose references", flush=True)
    normalized_dir = work_dir / "normalized-dose-edges"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_completed = 0
    for bucket in range(buckets):
        bucket_dir = dose_base_dir / f"dose_bucket={bucket}"
        if not bucket_dir.is_dir() or not any(bucket_dir.glob("*.parquet")):
            continue
        source_glob = _sql_literal((bucket_dir / "*.parquet").as_posix())
        normalized_path = _sql_literal(
            (normalized_dir / f"part-{bucket:03d}.parquet").as_posix()
        )
        connection.execute(
            f"""
            COPY (
        WITH candidates AS (
            SELECT e.*,
                   d.support AS drug_support,
                   d.median_log_amount AS drug_median,
                   d.iqr_log_amount AS drug_iqr,
                   d.p90_log_amount AS drug_p90,
                   d.daily_support AS drug_daily_support,
                   d.median_log_daily_amount AS drug_daily_median,
                   d.iqr_log_daily_amount AS drug_daily_iqr,
                   d.p90_log_daily_amount AS drug_daily_p90,
                   g.support AS dimension_support,
                   g.median_log_amount AS dimension_median,
                   g.iqr_log_amount AS dimension_iqr,
                   g.p90_log_amount AS dimension_p90,
                   g.daily_support AS dimension_daily_support,
                   g.median_log_daily_amount AS dimension_daily_median,
                   g.iqr_log_daily_amount AS dimension_daily_iqr,
                   g.p90_log_daily_amount AS dimension_daily_p90
            FROM read_parquet('{source_glob}', hive_partitioning = false) e
            LEFT JOIN dose_train_drug_stats d USING (dc_id, dose_dimension)
            LEFT JOIN dose_train_dimension_stats g USING (dose_dimension)
        ), reference_values AS (
            SELECT c.*,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_iqr > 1e-12 THEN 'drug'
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_support > 0 THEN 'dimension'
                   END AS amount_reference_level,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_iqr > 1e-12 THEN drug_support
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_support > 0 THEN dimension_support
                   END::BIGINT AS amount_reference_support,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_iqr > 1e-12 THEN drug_median
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_support > 0 THEN dimension_median
                   END::DOUBLE AS amount_median,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_iqr > 1e-12 THEN drug_iqr
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_support > 0 THEN
                                CASE WHEN dimension_iqr > 1e-12
                                     THEN dimension_iqr ELSE 1.0 END
                   END::DOUBLE AS amount_iqr,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_iqr > 1e-12 THEN drug_p90
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_support > 0 THEN dimension_p90
                   END::DOUBLE AS amount_p90,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_daily_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_daily_iqr > 1e-12 THEN 'drug'
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_daily_support > 0 THEN 'dimension'
                   END AS daily_reference_level,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_daily_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_daily_iqr > 1e-12 THEN drug_daily_support
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_daily_support > 0 THEN dimension_daily_support
                   END::BIGINT AS daily_reference_support,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_daily_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_daily_iqr > 1e-12 THEN drug_daily_median
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_daily_support > 0 THEN dimension_daily_median
                   END::DOUBLE AS daily_median,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_daily_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_daily_iqr > 1e-12 THEN drug_daily_iqr
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_daily_support > 0 THEN
                                CASE WHEN dimension_daily_iqr > 1e-12
                                     THEN dimension_daily_iqr ELSE 1.0 END
                   END::DOUBLE AS daily_iqr,
                   CASE
                       WHEN ingredient_normalization_eligible = 1
                            AND drug_daily_support >= {DRUG_DOSE_MIN_SUPPORT}
                            AND drug_daily_iqr > 1e-12 THEN drug_daily_p90
                       WHEN ingredient_normalization_eligible = 1
                            AND dimension_daily_support > 0 THEN dimension_daily_p90
                   END::DOUBLE AS daily_p90
            FROM candidates c
        )
        SELECT * EXCLUDE (
                   drug_support, drug_median, drug_iqr, drug_p90,
                   drug_daily_support, drug_daily_median, drug_daily_iqr,
                   drug_daily_p90, dimension_support, dimension_median,
                   dimension_iqr, dimension_p90, dimension_daily_support,
                   dimension_daily_median, dimension_daily_iqr,
                   dimension_daily_p90, amount_median, amount_iqr, amount_p90,
                   daily_median, daily_iqr, daily_p90
               ),
               CASE WHEN amount_reference_level IS NOT NULL THEN
                   least(
                       greatest(
                           (ln(canonical_amount) - amount_median) / amount_iqr,
                           -{RELATIVE_LOG_DOSE_CLIP}
                       ),
                       {RELATIVE_LOG_DOSE_CLIP}
                   )
               END::DOUBLE AS relative_log_amount,
               CASE WHEN amount_reference_level IS NOT NULL
                    THEN (ln(canonical_amount) > amount_p90)::INTEGER END
                   AS above_train_p90,
               CASE WHEN daily_reference_level IS NOT NULL THEN
                   least(
                       greatest(
                           (ln(canonical_daily_amount) - daily_median) / daily_iqr,
                           -{RELATIVE_LOG_DOSE_CLIP}
                       ),
                       {RELATIVE_LOG_DOSE_CLIP}
                   )
               END::DOUBLE AS relative_log_daily_amount,
               CASE WHEN daily_reference_level IS NOT NULL
                    THEN (ln(canonical_daily_amount) > daily_p90)::INTEGER END
                   AS above_train_daily_p90
        FROM reference_values
            ) TO '{normalized_path}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        shutil.rmtree(bucket_dir, ignore_errors=True)
        normalized_completed += 1
        if normalized_completed == 1 or normalized_completed % 8 == 0:
            print(
                f"    normalized dosage: {normalized_completed}/{buckets} shards",
                flush=True,
            )
    if not any(normalized_dir.glob("*.parquet")):
        raise TabularFeatureBuildError("dosage partitioning produced no normalized edges")
    normalized_glob = _sql_literal((normalized_dir / "*.parquet").as_posix())
    connection.execute(
        f"CREATE VIEW normalized_dose_edges AS "
        f"SELECT * FROM read_parquet('{normalized_glob}', union_by_name = true)"
    )
    shutil.rmtree(dose_base_dir, ignore_errors=True)
    print("  dosage: aggregating patient dosage features", flush=True)
    _create_bucketed_dosage_aggregates(
        connection,
        work_dir=work_dir,
        buckets=buckets,
    )


def _create_bucketed_dosage_aggregates(
    connection: duckdb.DuckDBPyConnection,
    *,
    work_dir: Path,
    buckets: int,
) -> None:
    source_dir = work_dir / "dosage-patient" / "source"
    output_dir = work_dir / "dosage-patient" / "aggregates"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
            SELECT primaryid, faers_raw, route, canonical_amount,
                   canonical_daily_amount, relative_log_amount,
                   relative_log_daily_amount, above_train_p90,
                   cast(hash(primaryid) % {buckets} AS INTEGER) AS bucket
            FROM normalized_dose_edges
        ) TO '{_sql_literal(source_dir.as_posix())}' (
            FORMAT PARQUET, COMPRESSION SNAPPY, PARTITION_BY (bucket),
            ROW_GROUP_SIZE 100000
        )
        """
    )
    completed = 0
    for bucket in range(buckets):
        bucket_dir = source_dir / f"bucket={bucket}"
        if not bucket_dir.is_dir() or not any(bucket_dir.glob("*.parquet")):
            continue
        source_glob = _sql_literal((bucket_dir / "*.parquet").as_posix())
        destination = _sql_literal((output_dir / f"part-{bucket:03d}.parquet").as_posix())
        connection.execute(
            f"""
            COPY (
                WITH route_flags AS (
                    SELECT e.*,
                           regexp_matches(
                               upper(trim(coalesce(route, ''))),
                               'INTRAVENOUS|(^|[^A-Z])IV([^A-Z]|$)'
                           ) AS has_iv,
                           regexp_matches(
                               upper(trim(coalesce(route, ''))),
                               'SUBCUTANEOUS|SUBCUT|(^|[^A-Z])SC([^A-Z]|$)'
                           ) AS has_sc,
                           regexp_matches(
                               upper(trim(coalesce(route, ''))),
                               'INTRAMUSCULAR|(^|[^A-Z])IM([^A-Z]|$)'
                           ) AS has_im,
                           regexp_matches(
                               upper(trim(coalesce(route, ''))),
                               'INTRAVENOUS|INTRAMUSCULAR|SUBCUTANEOUS|SUBCUT|INFUSION|PARENTERAL|(^|[^A-Z])(IV|IM|SC)([^A-Z]|$)'
                           ) AS is_parenteral
                    FROM read_parquet('{source_glob}') e
                ), patient_counts AS (
                    SELECT primaryid,
                           count(*)::BIGINT AS exposure_count,
                           count(*) FILTER (WHERE canonical_amount IS NOT NULL)::BIGINT
                               AS numeric_count,
                           count(*) FILTER (
                               WHERE canonical_daily_amount IS NOT NULL
                           )::BIGINT AS scheduled_count,
                           count(*) FILTER (
                               WHERE relative_log_amount IS NOT NULL
                           )::BIGINT AS relative_count,
                           count(*) FILTER (
                               WHERE relative_log_daily_amount IS NOT NULL
                           )::BIGINT AS daily_relative_count,
                           count(DISTINCT nullif(faers_raw, ''))::BIGINT AS drug_count,
                           count(DISTINCT nullif(faers_raw, '')) FILTER (
                               WHERE is_parenteral
                           )::BIGINT AS parenteral_drug_count
                    FROM route_flags GROUP BY primaryid
                )
                SELECT e.primaryid,
                       (p.numeric_count::DOUBLE / nullif(p.exposure_count, 0))::DOUBLE
                           AS dose_normalized_numeric_fraction,
                       (p.scheduled_count::DOUBLE / nullif(p.exposure_count, 0))::DOUBLE
                           AS dose_normalized_scheduled_fraction,
                       (p.relative_count::DOUBLE / nullif(p.exposure_count, 0))::DOUBLE
                           AS dose_normalized_relative_available_fraction,
                       coalesce(avg(e.relative_log_amount), 0.0)::DOUBLE
                           AS dose_normalized_relative_log_mean,
                       coalesce(max(e.relative_log_amount), 0.0)::DOUBLE
                           AS dose_normalized_relative_log_max,
                       (p.daily_relative_count::DOUBLE /
                           nullif(p.exposure_count, 0))::DOUBLE
                           AS dose_normalized_daily_relative_available_fraction,
                       coalesce(avg(e.relative_log_daily_amount), 0.0)::DOUBLE
                           AS dose_normalized_daily_relative_log_mean,
                       coalesce(max(e.relative_log_daily_amount), 0.0)::DOUBLE
                           AS dose_normalized_daily_relative_log_max,
                       count(*) FILTER (WHERE e.above_train_p90 = 1)::INTEGER
                           AS dose_normalized_above_train_p90_count,
                       coalesce(
                           count(*) FILTER (WHERE e.above_train_p90 = 1)::DOUBLE /
                               nullif(p.relative_count, 0),
                           0.0
                       )::DOUBLE AS dose_normalized_above_train_p90_fraction,
                       p.parenteral_drug_count::INTEGER
                           AS dose_normalized_num_parenteral_drugs,
                       coalesce(
                           p.parenteral_drug_count::DOUBLE /
                               nullif(p.drug_count, 0), 0.0
                       )::DOUBLE AS dose_normalized_parenteral_fraction,
                       max(e.has_iv)::INTEGER AS dose_normalized_has_iv,
                       max(e.has_sc)::INTEGER AS dose_normalized_has_sc,
                       max(e.has_im)::INTEGER AS dose_normalized_has_im,
                       {bucket}::INTEGER AS bucket
                FROM route_flags e JOIN patient_counts p USING (primaryid)
                GROUP BY e.primaryid, p.exposure_count, p.numeric_count,
                         p.scheduled_count, p.relative_count, p.daily_relative_count,
                         p.drug_count, p.parenteral_drug_count
            ) TO '{destination}' (
                FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000
            )
            """
        )
        completed += 1
        if completed == 1 or completed % 8 == 0:
            print(f"  dosage patient features: {completed}/{buckets} buckets", flush=True)
    integer_suffixes = ("_count", "_drugs", "_iv", "_sc", "_im")
    dosage_columns = "primaryid VARCHAR, " + ", ".join(
        f"{name} {'INTEGER' if name.endswith(integer_suffixes) else 'DOUBLE'}"
        for name in DOSAGE_PATIENT_FEATURES
    ) + ", bucket INTEGER"
    _create_parquet_union_view(
        connection,
        table_name="patient_dosage_aggregates",
        directory=output_dir,
        empty_columns=dosage_columns,
    )
    shutil.rmtree(source_dir, ignore_errors=True)


def _verify_enriched(path: Path, *, expected_rows: int) -> None:
    parquet = pq.ParquetFile(path)
    required = {
        "age_imputed_years",
        "age_missing",
        "sex_unknown",
        "weight_kg_normalized",
        "weight_missing",
        "max_ror",
        "mean_log_ror",
        "high_ror_count",
        "has_boxed_warning",
        "num_high_risk_atc",
        "atc_diversity",
        "therapeutic_duplicates",
        "atc_l2_diversity",
        "atc_l3_diversity",
        "atc_l4_diversity",
        "therapeutic_duplicates_l2",
        "num_high_risk_atc_groups",
        "exposure_route_diversity",
        "exposure_route_documented_fraction",
        "exposure_has_oral_route",
        "exposure_has_parenteral_route",
        "exposure_has_topical_route",
        "exposure_dose_documented_fraction",
        "exposure_dose_unit_diversity",
        "exposure_dose_form_diversity",
        "exposure_has_solid_form",
        "exposure_has_liquid_form",
        "exposure_has_injectable_form",
        "exposure_frequency_documented_fraction",
        "exposure_frequency_diversity",
        "age_group_0_17",
        "age_group_18_40",
        "age_group_41_64",
        "age_group_65_plus",
        "num_drugs_squared",
        "polypharmacy_age",
        *DOSAGE_PATIENT_FEATURES,
        *(f"atc_l1_count_{letter.lower()}" for letter in ATC_INITIALS),
    }
    if not required.issubset(parquet.schema_arrow.names):
        raise TabularFeatureBuildError("enriched cohort is missing requested feature columns")
    if parquet.metadata.num_rows != expected_rows:
        raise TabularFeatureBuildError("enriched cohort row count changed")
    for row_group in range(parquet.metadata.num_row_groups):
        for column in range(parquet.metadata.num_columns):
            if parquet.metadata.row_group(row_group).column(column).compression != "SNAPPY":
                raise TabularFeatureBuildError("enriched cohort contains non-Snappy columns")


def _validate_input_columns(*, cohort: Path, report_drug: Path) -> None:
    cohort_required = {
        "primaryid",
        "report_date",
        "age_years",
        "sex",
        "weight",
        "weight_unit",
        "num_drugs",
        "is_serious",
        "split",
    }
    edge_required = {
        "primaryid",
        "drugname",
        "route",
        "dose_amt",
        "dose_unit",
        "dose_form",
        "dose_freq",
    }
    for path, required in ((cohort, cohort_required), (report_drug, edge_required)):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(required - available)
        if missing:
            raise TabularFeatureBuildError(f"{path} is missing required columns: {missing}")


def _cohort_passthrough_sql(cohort: Path) -> str:
    generated = {"age_missing", "sex_unknown"}
    available = set(pq.ParquetFile(cohort).schema_arrow.names)
    excluded = sorted(generated & available)
    if not excluded:
        return "c.*"
    return f"c.* EXCLUDE ({', '.join(excluded)})"


def _verify_risk_lookup(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    expected = {"dc_id", "a", "b", "c", "d", "support", "ror"}
    if set(parquet.schema_arrow.names) != expected:
        raise TabularFeatureBuildError("invalid frozen drug-risk lookup schema")
    for row_group in range(parquet.metadata.num_row_groups):
        for column in range(parquet.metadata.num_columns):
            if parquet.metadata.row_group(row_group).column(column).compression != "SNAPPY":
                raise TabularFeatureBuildError("drug-risk lookup contains non-Snappy columns")


def _verify_dose_edges(path: Path, *, expected_rows: int) -> None:
    parquet = pq.ParquetFile(path)
    required = {
        "primaryid",
        "drug_seq",
        "drugname",
        "faers_raw",
        "dose_amt",
        "dose_unit",
        "dose_freq",
        "dictionary_positive_dc_id_count",
        "dc_id",
        "ingredient_normalization_eligible",
        "dose_dimension",
        "canonical_amount",
        "administrations_per_day",
        "canonical_daily_amount",
        "amount_reference_level",
        "amount_reference_support",
        "relative_log_amount",
        "above_train_p90",
        "daily_reference_level",
        "daily_reference_support",
        "relative_log_daily_amount",
    }
    if not required.issubset(parquet.schema_arrow.names):
        raise TabularFeatureBuildError("normalized dosage edges are missing audit columns")
    if parquet.metadata.num_rows != expected_rows:
        raise TabularFeatureBuildError("normalized dosage edge row count changed")
    _verify_snappy(parquet, label="normalized dosage edges")


def _verify_dose_lookup(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    required = {
        "reference_level",
        "dc_id",
        "dose_dimension",
        "support",
        "median_log_amount",
        "iqr_log_amount",
        "p90_log_amount",
        "daily_support",
        "median_log_daily_amount",
        "iqr_log_daily_amount",
        "p90_log_daily_amount",
        "fallback_support",
        "fallback_median_log_amount",
        "fallback_iqr_log_amount",
        "fallback_p90_log_amount",
        "fallback_daily_support",
        "fallback_median_log_daily_amount",
        "fallback_iqr_log_daily_amount",
        "fallback_p90_log_daily_amount",
    }
    if not required.issubset(parquet.schema_arrow.names):
        raise TabularFeatureBuildError("invalid dosage-normalization lookup schema")
    _verify_snappy(parquet, label="dosage-normalization lookup")


def _verify_snappy(parquet: pq.ParquetFile, *, label: str) -> None:
    for row_group in range(parquet.metadata.num_row_groups):
        for column in range(parquet.metadata.num_columns):
            if parquet.metadata.row_group(row_group).column(column).compression != "SNAPPY":
                raise TabularFeatureBuildError(f"{label} contains non-Snappy columns")


def _write_manifest(
    path: Path,
    *,
    record: TabularFeatureBuildRecord,
    inputs: dict[str, str],
    outputs: dict[str, str],
) -> None:
    payload = {
        "dataset": "TekaRx enriched cohort",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "input_sha256": inputs,
        "output_sha256": outputs,
        "ror_scope": DRUG_RISK_SCOPE,
        "dosage_normalization": {
            "scope": DOSE_NORMALIZATION_SCOPE,
            "drug_reference_minimum_support": DRUG_DOSE_MIN_SUPPORT,
            "relative_log_clip": RELATIVE_LOG_DOSE_CLIP,
            "combination_policy": (
                "relative ingredient dose unavailable when one FAERS raw name maps "
                "to multiple positive DrugCentral ids"
            ),
            "patient_features": list(DOSAGE_PATIENT_FEATURES),
        },
        "imputation_scope": "training-patient medians only",
        "prospective_exclusions": [
            "reactions",
            "reporter type",
            "drug role",
            "dechallenge/rechallenge",
            "outcomes as predictors",
        ],
        "record": asdict(record),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _sql_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'
