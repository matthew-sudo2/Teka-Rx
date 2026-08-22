"""Build train-frozen FAERS feature-rescue lookups and retrain the baseline."""

from __future__ import annotations

import json
import os
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
PAIR_RISK_SCOPE = (
    "strict-prior-date temporal encoding for fit-train patients; frozen full-fit-train "
    "lookup for later train/validation/test patients"
)

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
    processed = data_dir / "processed"
    base = processed / "tekarx_cohort_enriched.parquet"
    if not base.is_file():
        base = processed / "tekarx_cohort.parquet"
    dictionary = processed / "drug_dictionary.parquet"
    indi_files = sorted((data_dir / "interim" / "faers" / "indi").glob("*.parquet"))
    if not base.is_file() or not dictionary.is_file():
        raise FeatureRescueError("missing cohort or drug dictionary inputs")
    if not indi_files:
        raise FeatureRescueError(
            "missing staged FAERS INDI tables; run `tekarx build-faers --preset gnn-small "
            "--tables indi`"
        )

    output = processed / "tekarx_cohort_feature_rescue.parquet"
    pair_output = processed / "high_risk_drug_pairs.parquet"
    indication_output = processed / "indication_lookup.parquet"
    database = data_dir / "interim" / f".feature-rescue-{os.getpid()}.duckdb"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.unlink(missing_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    pair_temporary = pair_output.with_suffix(".parquet.tmp")
    indication_temporary = indication_output.with_suffix(".parquet.tmp")
    for path in (temporary, pair_temporary, indication_temporary):
        path.unlink(missing_ok=True)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(database))
        configure_duckdb(
            connection,
            data_dir=data_dir,
            stage="feature-rescue",
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
        )
        _copy_table(connection, "top_pair_lookup", pair_temporary)
        _copy_table(connection, "indication_lookup", indication_temporary)
        base_sql = _sql_literal(base.as_posix())
        output_sql = _sql_literal(temporary.as_posix())
        indication_hash_select = "".join(
            f", coalesce(h.indication_hash_{index:02d}, 0)::INTEGER "
            f"AS indication_hash_{index:02d}"
            for index in range(INDICATION_HASH_BINS)
        )
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
                FROM read_parquet('{base_sql}') c
                LEFT JOIN patient_indications i USING (primaryid)
                LEFT JOIN patient_indication_hash h USING (primaryid)
                LEFT JOIN patient_pair_features p USING (primaryid)
                ORDER BY c.report_date, c.primaryid
            ) TO '{output_sql}'
            (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000)
            """
        )
        expected_rows = connection.execute(
            f"SELECT count(*) FROM read_parquet('{base_sql}')"
        ).fetchone()[0]
    except BaseException:
        for path in (temporary, pair_temporary, indication_temporary):
            path.unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:
            connection.close()
        database.unlink(missing_ok=True)

    _verify_output(temporary, expected_rows=expected_rows)
    os.replace(temporary, output)
    os.replace(pair_temporary, pair_output)
    os.replace(indication_temporary, indication_output)
    graph_record: GraphBuildRecord | None = None
    if rebuild_graph:
        graph_record = build_graph(
            data_dir=data_dir,
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
) -> tuple[object, object, object]:
    base_sql = _sql_literal(base.as_posix())
    dictionary_sql = _sql_literal(dictionary.as_posix())
    indi_sql = _sql_literal(_glob_path(indi_files))
    connection.execute(
        f"""
        CREATE TABLE training_patients AS
        SELECT primaryid, is_serious, report_date
        FROM read_parquet('{base_sql}')
        WHERE split = 'train' AND year(report_date) <= {int(training_end_year)}
        """
    )
    coverage = connection.execute(
        "SELECT count(*), min(report_date), max(report_date) FROM training_patients"
    ).fetchone()
    if not coverage or not coverage[0]:
        raise FeatureRescueError("no training reports satisfy the temporal cutoff")

    connection.execute(
        f"""
        CREATE TABLE patient_drugs AS
        WITH raw_drugs AS (
            SELECT c.primaryid, trim(value) AS faers_raw
            FROM read_parquet('{base_sql}') c,
                 unnest(string_split(c.drug_list_str, '|')) names(value)
            WHERE trim(value) <> ''
        )
        SELECT DISTINCT r.primaryid, d.dc_id::BIGINT AS dc_id
        FROM raw_drugs r JOIN read_parquet('{dictionary_sql}') d USING (faers_raw)
        WHERE d.dc_id > 0
        """
    )
    connection.execute(
        """
        CREATE TABLE patient_pairs AS
        SELECT DISTINCT a.primaryid, a.dc_id AS dc_id_1, b.dc_id AS dc_id_2
        FROM patient_drugs a
        JOIN patient_drugs b ON a.primaryid = b.primaryid AND a.dc_id < b.dc_id
        """
    )
    connection.execute(
        f"""
        CREATE TABLE full_pair_risk AS
        WITH totals AS (
            SELECT count(*) FILTER (WHERE is_serious = 1) AS total_serious,
                   count(*) FILTER (WHERE is_serious = 0) AS total_nonserious
            FROM training_patients
        ), counts AS (
            SELECT p.dc_id_1, p.dc_id_2,
                   count(*) FILTER (WHERE t.is_serious = 1) AS a,
                   count(*) FILTER (WHERE t.is_serious = 0) AS b
            FROM patient_pairs p JOIN training_patients t USING (primaryid)
            GROUP BY p.dc_id_1, p.dc_id_2
        )
        SELECT dc_id_1, dc_id_2, a::BIGINT AS a, b::BIGINT AS b,
               (total_serious - a)::BIGINT AS c,
               (total_nonserious - b)::BIGINT AS d,
               (a + b)::BIGINT AS pair_reports,
               (a::DOUBLE / nullif(a + b, 0)) /
               nullif((total_serious - a)::DOUBLE /
                      nullif(total_serious + total_nonserious - a - b, 0), 0) AS prr,
               ln(
                   ((a + 0.5)::DOUBLE * (total_nonserious - b + 0.5)::DOUBLE) /
                   ((b + 0.5)::DOUBLE * (total_serious - a + 0.5)::DOUBLE)
               )::DOUBLE AS log_ror
        FROM counts CROSS JOIN totals
        WHERE a + b >= {int(minimum_pair_reports)}
          AND total_serious > 0 AND total_nonserious > 0
        """
    )
    connection.execute(
        f"""
        CREATE TABLE top_pair_lookup AS
        SELECT dc_id_1, dc_id_2, a, b, c, d, pair_reports, prr, log_ror,
               row_number() OVER (ORDER BY prr DESC, pair_reports DESC, dc_id_1, dc_id_2)::INTEGER
                   AS risk_rank
        FROM full_pair_risk WHERE prr IS NOT NULL AND isfinite(prr)
        ORDER BY risk_rank LIMIT {int(top_pairs)}
        """
    )
    connection.execute(
        """
        CREATE TABLE training_totals_daily AS
        SELECT report_date,
               count(*) FILTER (WHERE is_serious = 1)::BIGINT AS serious_on_date,
               count(*) FILTER (WHERE is_serious = 0)::BIGINT AS nonserious_on_date
        FROM training_patients WHERE report_date IS NOT NULL GROUP BY report_date
        """
    )
    connection.execute(
        """
        CREATE TABLE training_total_history AS
        SELECT report_date,
               sum(serious_on_date) OVER (
                   ORDER BY report_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS total_serious,
               sum(nonserious_on_date) OVER (
                   ORDER BY report_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS total_nonserious
        FROM training_totals_daily
        """
    )
    connection.execute(
        """
        CREATE TABLE training_pair_daily AS
        SELECT p.dc_id_1, p.dc_id_2, t.report_date,
               count(*) FILTER (WHERE t.is_serious = 1)::BIGINT AS a_on_date,
               count(*) FILTER (WHERE t.is_serious = 0)::BIGINT AS b_on_date
        FROM patient_pairs p JOIN training_patients t USING (primaryid)
        WHERE t.report_date IS NOT NULL
        GROUP BY p.dc_id_1, p.dc_id_2, t.report_date
        """
    )
    connection.execute(
        """
        CREATE TABLE training_pair_history AS
        SELECT dc_id_1, dc_id_2, report_date,
               sum(a_on_date) OVER (
                   PARTITION BY dc_id_1, dc_id_2 ORDER BY report_date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS a,
               sum(b_on_date) OVER (
                   PARTITION BY dc_id_1, dc_id_2 ORDER BY report_date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               )::BIGINT AS b
        FROM training_pair_daily
        """
    )
    connection.execute(
        f"""
        CREATE TABLE temporal_train_pair_scores AS
        WITH exposures AS (
            SELECT p.primaryid, p.dc_id_1, p.dc_id_2, t.report_date
            FROM patient_pairs p JOIN training_patients t USING (primaryid)
        ), with_totals AS (
            SELECT e.primaryid, e.dc_id_1, e.dc_id_2, e.report_date,
                   t.total_serious, t.total_nonserious
            FROM exposures e
            ASOF LEFT JOIN training_total_history t ON e.report_date > t.report_date
        )
        SELECT e.primaryid, e.dc_id_1, e.dc_id_2,
               CASE
                   WHEN h.dc_id_1 IS NULL OR h.a + h.b < {int(minimum_pair_reports)}
                        OR coalesce(e.total_serious, 0) = 0
                        OR coalesce(e.total_nonserious, 0) = 0 THEN NULL
                   ELSE ln(
                       ((h.a + 0.5)::DOUBLE *
                        (e.total_nonserious - h.b + 0.5)::DOUBLE) /
                       ((h.b + 0.5)::DOUBLE *
                        (e.total_serious - h.a + 0.5)::DOUBLE)
                   )::DOUBLE
               END AS log_ror
        FROM with_totals e
        ASOF LEFT JOIN training_pair_history h
          ON e.dc_id_1 = h.dc_id_1 AND e.dc_id_2 = h.dc_id_2
         AND e.report_date > h.report_date
        """
    )
    connection.execute(
        """
        CREATE TABLE patient_pair_scores AS
        SELECT primaryid, dc_id_1, dc_id_2, log_ror
        FROM temporal_train_pair_scores WHERE log_ror IS NOT NULL
        UNION ALL
        SELECT p.primaryid, p.dc_id_1, p.dc_id_2, f.log_ror
        FROM patient_pairs p
        JOIN full_pair_risk f USING (dc_id_1, dc_id_2)
        WHERE NOT EXISTS (
            SELECT 1 FROM training_patients t WHERE t.primaryid = p.primaryid
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE patient_pair_features AS
        SELECT primaryid,
               count(*) FILTER (WHERE log_ror > ln({PAIR_HIGH_RISK_ROR}))::INTEGER
                   AS num_high_risk_pairs,
               max(log_ror)::DOUBLE AS max_pair_log_ror,
               avg(log_ror)::DOUBLE AS mean_pair_log_ror,
               count(*)::INTEGER AS scored_pair_count
        FROM patient_pair_scores GROUP BY primaryid
        """
    )
    connection.execute(
        f"""
        CREATE TABLE indication_lookup AS
        SELECT DISTINCT upper(trim(i.indi_pt)) AS indication,
               mod(hash(upper(trim(i.indi_pt))), {INDICATION_HASH_BINS})::INTEGER AS hash_bin,
               regexp_matches(upper(trim(i.indi_pt)), '{MALIGNANCY_PATTERN}')::INTEGER
                   AS has_malignancy,
               regexp_matches(upper(trim(i.indi_pt)), '{CARDIO_PATTERN}')::INTEGER AS has_cardio,
               regexp_matches(upper(trim(i.indi_pt)), '{INFECTION_PATTERN}')::INTEGER
                   AS has_infection
        FROM read_parquet('{indi_sql}') i
        JOIN training_patients t USING (primaryid)
        WHERE i.indi_pt IS NOT NULL AND trim(i.indi_pt) <> ''
        """
    )
    hash_counts = ",\n".join(
        f"count(DISTINCT l.indication) FILTER (WHERE l.hash_bin = {index})::INTEGER "
        f"AS indication_hash_{index:02d}"
        for index in range(INDICATION_HASH_BINS)
    )
    connection.execute(
        f"""
        CREATE TABLE patient_indications AS
        SELECT i.primaryid,
               max(l.has_malignancy)::INTEGER AS has_malignancy,
               max(l.has_cardio)::INTEGER AS has_cardio,
               max(l.has_infection)::INTEGER AS has_infection
        FROM read_parquet('{indi_sql}') i
        JOIN indication_lookup l ON l.indication = upper(trim(i.indi_pt))
        JOIN read_parquet('{base_sql}') c USING (primaryid)
        GROUP BY i.primaryid
        """
    )
    connection.execute(
        f"""
        CREATE TABLE patient_indication_hash AS
        SELECT i.primaryid, {hash_counts}
        FROM read_parquet('{indi_sql}') i
        JOIN indication_lookup l ON l.indication = upper(trim(i.indi_pt))
        JOIN read_parquet('{base_sql}') c USING (primaryid)
        GROUP BY i.primaryid
        """
    )
    return coverage


def _copy_table(connection: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    connection.execute(
        f"COPY (SELECT * FROM {table}) TO '{_sql_literal(path.as_posix())}' "
        "(FORMAT PARQUET, COMPRESSION SNAPPY)"
    )


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
    if parquet.metadata.num_rows != expected_rows or not required.issubset(
        parquet.schema_arrow.names
    ):
        raise FeatureRescueError("invalid feature-rescue cohort schema or row count")


def _glob_path(files: list[Path]) -> str:
    parents = {path.parent for path in files}
    if len(parents) != 1:
        raise FeatureRescueError("FAERS staged files do not share one table directory")
    return (next(iter(parents)) / "*.parquet").as_posix()


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
        },
        "indication_encoding": {
            "vocabulary": "exact terms observed in fit-train only",
            "hash_bins": INDICATION_HASH_BINS,
            "held_out_unknown_terms": "ignored",
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
