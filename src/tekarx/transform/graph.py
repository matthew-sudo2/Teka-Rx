"""Build the TekaRx heterogeneous graph and a tabular XGBoost baseline."""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq

from tekarx.transform.dosage import DOSAGE_PATIENT_FEATURES
from tekarx.transform.duckdb_runtime import configure_duckdb
from tekarx.transform.graph_storage import (
    MEMMAP_GRAPH_FORMAT,
    MEMMAP_GRAPH_VERSION,
    GraphArrayBundle,
    load_graph_arrays,
)

ATC_INITIALS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DRUG_FEATURE_NAMES = (
    "zscore(log(train-only ROR))",
    "has_boxed_warning",
    *(f"ATC initial {letter}" for letter in ATC_INITIALS),
)
NORMALIZED_DOSAGE_FEATURE_PREFIX = "dose_normalized_"
NORMALIZED_DOSAGE_PATIENT_FEATURES = DOSAGE_PATIENT_FEATURES
BASE_PATIENT_FEATURES = ("age_over_120", "sex_binary", "num_drugs_over_50")
ENRICHED_PATIENT_FEATURES = (
    "max_ror",
    "high_ror_count",
    "has_boxed_warning",
    "num_high_risk_atc",
    "atc_diversity",
    "therapeutic_duplicates",
    "age_group_0_17",
    "age_group_18_40",
    "age_group_41_64",
    "age_group_65_plus",
    "num_drugs_squared",
    "polypharmacy_age",
    "has_malignancy",
    "has_cardio",
    "has_infection",
    "num_high_risk_pairs",
)
AUXILIARY_TARGETS = ("is_death", "is_hospitalization")
PROSPECTIVE_PATIENT_FEATURES = (
    "age_missing",
    "sex_unknown",
    "weight_kg_normalized",
    "weight_missing",
    "mean_log_ror",
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
    "max_pair_log_ror",
    "mean_pair_log_ror",
    "scored_pair_count",
)


class GraphBuildError(RuntimeError):
    """Raised when graph inputs or invariants are invalid."""


@dataclass(frozen=True)
class GraphBuildRecord:
    """Summary for one graph and baseline build."""

    graph_path: str
    tabular_path: str
    xgboost_model_path: str
    patient_nodes: int
    drug_nodes: int
    patient_drug_edges: int
    patient_feature_count: int
    mapped_drug_nodes: int
    frequent_unknown_nodes: int
    has_other_node: bool
    validation_auc: float
    storage_format: str
    graph_array_manifest_path: str
    numeric_storage_bytes: int
    materialization_batch_size: int
    edge_index_dtype: str


def build_graph(
    *,
    data_dir: Path,
    top_unknown: int = 500,
    memory_limit: str = "4GB",
    threads: int | None = None,
    xgb_rounds: int = 500,
    xgb_early_stopping: int = 30,
    xgb_max_depth: int = 5,
    xgb_max_leaves: int = 0,
    cohort_path: Path | None = None,
    storage_mode: str = "memory-mapped",
    materialization_batch_size: int = 131_072,
    xgb_batch_size: int = 65_536,
) -> GraphBuildRecord:
    """Build the graph, tabular arrays, and a temporal XGBoost baseline.

    The default format stores every large numeric array once in a page-backed
    ``.npy`` file.  DuckDB results are copied into those arrays in bounded Arrow
    record batches; no full Arrow table, Python row list, or edge-sized split
    string array is created.  ``storage_mode='legacy'`` remains available for
    consumers that require a single PyG ``HeteroData`` pickle.
    """
    if top_unknown < 0:
        raise ValueError("top_unknown cannot be negative")
    if xgb_rounds < 1 or xgb_early_stopping < 1:
        raise ValueError("XGBoost rounds and early stopping must be positive")
    if storage_mode not in {"memory-mapped", "legacy"}:
        raise ValueError("storage_mode must be 'memory-mapped' or 'legacy'")
    if materialization_batch_size < 1 or xgb_batch_size < 1:
        raise ValueError("materialization and XGBoost batch sizes must be positive")
    try:
        import torch
        import xgboost as xgb
    except ImportError as exc:
        raise GraphBuildError(
            'missing graph dependencies; install with python -m pip install -e ".[graph]"'
        ) from exc

    paths = _required_paths(data_dir, cohort_path=cohort_path)
    cohort_columns = set(pq.ParquetFile(paths["cohort"]).schema_arrow.names)
    extra_features = _discover_patient_features(cohort_columns)
    age_one_hot = {
        "age_group_0_17",
        "age_group_18_40",
        "age_group_41_64",
        "age_group_65_plus",
    }
    present_age_groups = age_one_hot & cohort_columns
    if present_age_groups and present_age_groups != age_one_hot:
        raise GraphBuildError(
            f"partially encoded age groups are missing: {sorted(age_one_hot - cohort_columns)}"
        )
    auxiliary_targets = tuple(name for name in AUXILIARY_TARGETS if name in cohort_columns)
    has_explicit_imputed_age = "age_imputed_years" in cohort_columns
    base_feature_names = (
        (
            "age_imputed_years_over_120",
            "sex_binary",
            "num_drugs_over_50",
        )
        if has_explicit_imputed_age
        else BASE_PATIENT_FEATURES
    )
    feature_names = (*base_feature_names, *extra_features)
    if set(feature_names) & set(AUXILIARY_TARGETS):
        raise GraphBuildError("auxiliary outcomes cannot be patient input features")
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    database = data_dir / "interim" / f".graph-{os.getpid()}.duckdb"
    database.unlink(missing_ok=True)
    connection: duckdb.DuckDBPyConnection | None = None
    materialization: dict[str, Any] | None = None
    try:
        connection = duckdb.connect(str(database))
        configure_duckdb(
            connection,
            data_dir=data_dir,
            stage="graph",
            memory_limit=memory_limit,
            threads=threads,
        )
        _prepare_graph_tables(
            connection,
            paths=paths,
            top_unknown=top_unknown,
            extra_patient_columns=(
                *(("age_imputed_years",) if has_explicit_imputed_age else ()),
                *extra_features,
                *auxiliary_targets,
            ),
        )
        stats = dict(
            connection.execute(
                """
                SELECT 'mapped' AS kind, count(*)::BIGINT AS value
                FROM drug_nodes WHERE node_kind = 'mapped'
                UNION ALL SELECT 'frequent_unknown', count(*)::BIGINT
                FROM drug_nodes WHERE node_kind = 'frequent_unknown'
                UNION ALL SELECT 'other', count(*)::BIGINT
                FROM drug_nodes WHERE node_kind = 'other'
                """
            ).fetchall()
        )
        materialization = _materialize_graph_arrays(
            connection,
            output_dir=output_dir,
            paths=paths,
            feature_names=feature_names,
            extra_features=extra_features,
            auxiliary_targets=auxiliary_targets,
            has_explicit_imputed_age=has_explicit_imputed_age,
            batch_size=materialization_batch_size,
        )
    finally:
        if connection is not None:
            connection.close()
        database.unlink(missing_ok=True)
    if materialization is None:  # pragma: no cover - defensive
        raise GraphBuildError("graph materialization did not complete")
    graph_path = output_dir / "tekarx_graph.pt"
    array_manifest_path = Path(materialization["manifest_path"])
    _write_graph_descriptor(graph_path, array_manifest_path=array_manifest_path, torch=torch)
    bundle = load_graph_arrays(
        graph_path, mmap_mode="r+" if storage_mode == "legacy" else "r"
    )
    if storage_mode == "legacy":
        _write_legacy_graph(
            graph_path,
            bundle=bundle,
            auxiliary_targets=auxiliary_targets,
            torch=torch,
        )

    patient_x = bundle.arrays["patient_x"]
    labels = bundle.arrays["patient_y"]
    split_ids = bundle.arrays["patient_split_id"]
    train_mask = split_ids == 0
    validation_mask = split_ids == 1
    test_mask = split_ids == 2
    _verify_masks(train_mask, validation_mask, test_mask)

    tabular_path = output_dir / "tekarx_tabular_baseline.npz"
    tabular_temp = tabular_path.with_suffix(".npz.tmp")
    with tabular_temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            X=patient_x,
            y=labels,
            primaryid=bundle.arrays["patient_primaryid"],
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            feature_names=np.asarray(feature_names),
        )
    os.replace(tabular_temp, tabular_path)

    train_matrix = _quantile_matrix_for_split(
        xgb,
        patient_x,
        labels,
        split_ids,
        split_id=0,
        batch_size=xgb_batch_size,
    )
    validation_matrix = _quantile_matrix_for_split(
        xgb,
        patient_x,
        labels,
        split_ids,
        split_id=1,
        batch_size=xgb_batch_size,
        reference=train_matrix,
    )
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "max_depth": xgb_max_depth,
            "max_leaves": xgb_max_leaves,
            "grow_policy": "lossguide" if xgb_max_leaves else "depthwise",
            "eta": 0.03 if xgb_max_leaves else 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "seed": 42,
            "nthread": threads or 0,
            "max_bin": 256,
        },
        train_matrix,
        num_boost_round=xgb_rounds,
        evals=[(validation_matrix, "validation")],
        early_stopping_rounds=xgb_early_stopping,
        verbose_eval=False,
    )
    validation_scores = booster.predict(validation_matrix)
    validation_auc = binary_auc(np.asarray(labels[validation_mask]), validation_scores)
    model_path = output_dir / "xgboost_baseline.json"
    model_temp = model_path.with_name("xgboost_baseline.tmp.json")
    booster.save_model(model_temp)
    os.replace(model_temp, model_path)

    record = GraphBuildRecord(
        graph_path=str(graph_path),
        tabular_path=str(tabular_path),
        xgboost_model_path=str(model_path),
        patient_nodes=int(materialization["patient_nodes"]),
        drug_nodes=int(materialization["drug_nodes"]),
        patient_drug_edges=int(materialization["patient_drug_edges"]),
        patient_feature_count=patient_x.shape[1],
        mapped_drug_nodes=int(stats.get("mapped", 0)),
        frequent_unknown_nodes=int(stats.get("frequent_unknown", 0)),
        has_other_node=bool(stats.get("other", 0)),
        validation_auc=validation_auc,
        storage_format=(MEMMAP_GRAPH_FORMAT if storage_mode == "memory-mapped" else "pyg"),
        graph_array_manifest_path=str(array_manifest_path),
        numeric_storage_bytes=int(materialization["numeric_storage_bytes"]),
        materialization_batch_size=materialization_batch_size,
        edge_index_dtype=str(bundle.arrays["edge_patient_index"].dtype),
    )
    print(f"XGBoost validation AUC: {validation_auc:.6f}")
    _write_manifest(
        output_dir / "graph_manifest.json",
        record,
        feature_names=feature_names,
        array_manifest=bundle.manifest,
    )
    return record


def _materialize_graph_arrays(
    connection: duckdb.DuckDBPyConnection,
    *,
    output_dir: Path,
    paths: dict[str, Path],
    feature_names: tuple[str, ...],
    extra_features: tuple[str, ...],
    auxiliary_targets: tuple[str, ...],
    has_explicit_imputed_age: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Stream DuckDB graph tables into one-copy numeric NPY artifacts."""
    array_dir = output_dir / "tekarx_graph_arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = array_dir / "manifest.json"
    previous_artifacts = _manifest_artifacts(manifest_path)
    build_id = f"{time.time_ns()}-{os.getpid()}"
    created: list[Path] = []
    arrays: dict[str, dict[str, Any]] = {}
    try:
        patient_count = int(connection.execute("SELECT count(*) FROM patients").fetchone()[0])
        drug_count = int(connection.execute("SELECT count(*) FROM drug_nodes").fetchone()[0])
        edge_count = int(connection.execute("SELECT count(*) FROM graph_edges").fetchone()[0])
        if min(patient_count, drug_count, edge_count) < 1:
            raise GraphBuildError("graph must contain patients, drugs, and exposure edges")

        invalid_primaryids = int(
            connection.execute(
                "SELECT count(*) FROM patients WHERE try_cast(primaryid AS BIGINT) IS NULL"
            ).fetchone()[0]
        )
        if invalid_primaryids:
            raise GraphBuildError(
                f"graph contains {invalid_primaryids} primaryid values that are not int64"
            )
        age_column = "age_imputed_years" if has_explicit_imputed_age else "age"
        train_age_median = connection.execute(
            f"""
            SELECT median(try_cast({_sql_identifier(age_column)} AS DOUBLE))
            FROM patients
            WHERE split = 'train'
              AND isfinite(try_cast({_sql_identifier(age_column)} AS DOUBLE))
            """
        ).fetchone()[0]
        train_age_median = float(train_age_median) if train_age_median is not None else 0.0

        feature_expressions = [
            (
                "least(greatest(coalesce(try_cast("
                f"{_sql_identifier(age_column)} AS DOUBLE), {train_age_median!r}), 0.0), "
                "120.0) / 120.0"
            ),
            "CASE WHEN upper(trim(coalesce(sex, ''))) = 'M' THEN 1.0 ELSE 0.0 END",
            "least(greatest(coalesce(try_cast(num_drugs AS DOUBLE), 0.0), 0.0), 50.0) / 50.0",
            *(
                f"coalesce(try_cast({_sql_identifier(name)} AS DOUBLE), 0.0)"
                for name in extra_features
            ),
        ]
        if len(feature_expressions) != len(feature_names):
            raise GraphBuildError("patient feature names and SQL expressions are misaligned")
        patient_select = [
            "try_cast(primaryid AS BIGINT) AS patient_primaryid",
            "coalesce(try_cast(is_serious AS TINYINT), 0) AS patient_y",
            "CASE split WHEN 'train' THEN 0 WHEN 'validation' THEN 1 "
            "WHEN 'test' THEN 2 ELSE -1 END::TINYINT AS patient_split_id",
            *(
                f"{expression}::FLOAT AS feature_{index:04d}"
                for index, expression in enumerate(feature_expressions)
            ),
            *(
                f"coalesce(try_cast({_sql_identifier(name)} AS TINYINT), 0) "
                f"AS auxiliary_{index:04d}"
                for index, name in enumerate(auxiliary_targets)
            ),
        ]
        patient_specs: list[tuple[str, str, np.dtype[Any], tuple[int, ...]]] = [
            ("patient_primaryid", "patient_primaryid", np.dtype("int64"), (patient_count,)),
            ("patient_y", "patient_y", np.dtype("int8"), (patient_count,)),
            ("patient_split_id", "patient_split_id", np.dtype("int8"), (patient_count,)),
            *(
                (
                    f"auxiliary_{index:04d}",
                    f"patient_{name}",
                    np.dtype("int8"),
                    (patient_count,),
                )
                for index, name in enumerate(auxiliary_targets)
            ),
        ]
        patient_query = (
            "SELECT " + ", ".join(patient_select) + " FROM patients ORDER BY patient_index"
        )
        patient_arrays = _stream_patient_query(
            connection,
            query=patient_query,
            row_count=patient_count,
            feature_count=len(feature_names),
            scalar_specs=patient_specs,
            output_dir=array_dir,
            build_id=build_id,
            batch_size=batch_size,
            created=created,
        )
        arrays.update(patient_arrays)

        ror_mean, ror_std = connection.execute(
            """
            SELECT coalesce(avg(ln(greatest(ror, 1e-12))), 0.0),
                   coalesce(stddev_pop(ln(greatest(ror, 1e-12))), 1.0)
            FROM drug_nodes WHERE ror IS NOT NULL
            """
        ).fetchone()
        ror_mean = float(ror_mean or 0.0)
        ror_std = float(ror_std or 1.0)
        if ror_std <= 0 or not math.isfinite(ror_std):
            ror_std = 1.0
        drug_expressions = [
            f"coalesce((ln(greatest(ror, 1e-12)) - {ror_mean!r}) / {ror_std!r}, 0.0)",
            "coalesce(try_cast(has_boxed_warning AS DOUBLE), 0.0)",
            *(
                "CASE WHEN regexp_matches(upper(coalesce(atc_code, '')), "
                f"'(^|[|]){letter}') THEN 1.0 ELSE 0.0 END"
                for letter in ATC_INITIALS
            ),
        ]
        drug_query = (
            "SELECT semantic_id::BIGINT AS drug_node_id, "
            + ", ".join(
                f"{expression}::FLOAT AS feature_{index:04d}"
                for index, expression in enumerate(drug_expressions)
            )
            + " FROM drug_nodes ORDER BY node_index"
        )
        drug_arrays = _stream_feature_query(
            connection,
            query=drug_query,
            row_count=drug_count,
            feature_count=len(DRUG_FEATURE_NAMES),
            matrix_name="drug_x",
            scalar_specs=[("drug_node_id", "drug_node_id", np.dtype("int64"), (drug_count,))],
            output_dir=array_dir,
            build_id=build_id,
            batch_size=batch_size,
            created=created,
        )
        arrays.update(drug_arrays)

        largest_index = max(patient_count - 1, drug_count - 1)
        index_dtype = np.dtype("int32" if largest_index <= np.iinfo(np.int32).max else "int64")
        sql_index_type = "INTEGER" if index_dtype == np.dtype("int32") else "BIGINT"
        edge_counts = {
            split: int(count)
            for split, count in connection.execute(
                """
                SELECT p.split, count(*)::BIGINT
                FROM graph_edges e JOIN patients p USING (patient_index)
                GROUP BY p.split
                """
            ).fetchall()
        }
        if set(edge_counts) != {"train", "validation", "test"}:
            raise GraphBuildError("every patient split must contribute graph edges")
        edge_query = f"""
            SELECT e.patient_index::{sql_index_type} AS edge_patient_index,
                   e.node_index::{sql_index_type} AS edge_drug_index
            FROM graph_edges e JOIN patients p USING (patient_index)
            ORDER BY CASE p.split WHEN 'train' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END,
                     e.patient_index, e.node_index
        """
        edge_arrays = _stream_scalar_query(
            connection,
            query=edge_query,
            row_count=edge_count,
            specs=[
                ("edge_patient_index", "edge_patient_index", index_dtype, (edge_count,)),
                ("edge_drug_index", "edge_drug_index", index_dtype, (edge_count,)),
            ],
            output_dir=array_dir,
            build_id=build_id,
            batch_size=batch_size,
            created=created,
        )
        arrays.update(edge_arrays)

        metadata_name = f"drug_nodes-{build_id}.parquet"
        metadata_path = array_dir / metadata_name
        connection.execute(
            f"""
            COPY (
                SELECT node_index, semantic_id, node_label, node_kind
                FROM drug_nodes ORDER BY node_index
            ) TO '{_sql_literal(metadata_path.as_posix())}'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
            """
        )
        created.append(metadata_path)

        split_offsets: dict[str, list[int]] = {}
        offset = 0
        for split in ("train", "validation", "test"):
            stop = offset + edge_counts[split]
            split_offsets[split] = [offset, stop]
            offset = stop
        if offset != edge_count:
            raise GraphBuildError("edge split offsets do not cover every edge")
        numeric_bytes = sum(int(item["nbytes"]) for item in arrays.values())
        manifest = {
            "format": MEMMAP_GRAPH_FORMAT,
            "format_version": MEMMAP_GRAPH_VERSION,
            "created_at_utc": _utc_now(),
            "patient_feature_names": list(feature_names),
            "drug_feature_names": list(DRUG_FEATURE_NAMES),
            "auxiliary_targets": list(auxiliary_targets),
            "arrays": arrays,
            "drug_metadata_path": metadata_name,
            "edge_order": {
                "sort": ["patient_split_id", "patient_index", "drug_index"],
                "split_offsets": split_offsets,
                "canonical_direction": "patient_to_drug",
                "reverse_direction": "swap the two index arrays while consuming",
            },
            "inductive_protocol": {
                "unknown_vocabulary": "train-only frequency; unseen held-out names map to OTHER",
                "patient_to_drug": "training edges selected by patient_split_id == 0",
                "drug_to_patient": "all canonical edges, reversed while consuming",
                "held_out_patient_messages_to_shared_drugs": False,
                "auxiliary_targets_in_x": False,
            },
            "materialization": {
                "record_batch_rows": batch_size,
                "full_arrow_tables": False,
                "python_row_lists": False,
                "canonical_edge_copies": 1,
                "edge_split_string_array": False,
                "edge_boolean_masks_stored": False,
                "patient_boolean_masks_stored": False,
                "index_dtype": str(index_dtype),
                "numeric_storage_bytes": numeric_bytes,
                "xgboost_input": "streamed QuantileDMatrix batches",
            },
            "counts": {
                "patient_nodes": patient_count,
                "drug_nodes": drug_count,
                "patient_drug_edges": edge_count,
                "edges_by_split": edge_counts,
            },
            "input_provenance": {
                name: _artifact_provenance(path) for name, path in paths.items()
            },
        }
        temporary_manifest = array_dir / f".manifest-{build_id}.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
        current_artifacts = {path.resolve() for path in created}
        for old_path in previous_artifacts - current_artifacts:
            with suppress(OSError):
                old_path.unlink()
        return {
            "manifest_path": str(manifest_path),
            "patient_nodes": patient_count,
            "drug_nodes": drug_count,
            "patient_drug_edges": edge_count,
            "numeric_storage_bytes": numeric_bytes,
        }
    except Exception:
        for path in created:
            with suppress(OSError):
                path.unlink(missing_ok=True)
        raise


def _stream_patient_query(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    row_count: int,
    feature_count: int,
    scalar_specs: list[tuple[str, str, np.dtype[Any], tuple[int, ...]]],
    output_dir: Path,
    build_id: str,
    batch_size: int,
    created: list[Path],
) -> dict[str, dict[str, Any]]:
    return _stream_feature_query(
        connection,
        query=query,
        row_count=row_count,
        feature_count=feature_count,
        matrix_name="patient_x",
        scalar_specs=scalar_specs,
        output_dir=output_dir,
        build_id=build_id,
        batch_size=batch_size,
        created=created,
    )


def _arrow_record_batch_reader(
    result: duckdb.DuckDBPyConnection,
    batch_size: int,
) -> Any:
    """Return a streaming Arrow reader across supported DuckDB releases."""
    to_arrow_reader = getattr(result, "to_arrow_reader", None)
    if callable(to_arrow_reader):
        return to_arrow_reader(batch_size)

    # Older supported DuckDB releases expose the same stream through this name.
    fetch_record_batch = getattr(result, "fetch_record_batch", None)
    if callable(fetch_record_batch):
        return fetch_record_batch(batch_size)

    raise GraphBuildError(
        "installed DuckDB cannot stream query results as Arrow record batches; "
        "expected to_arrow_reader() or fetch_record_batch()"
    )


def _stream_feature_query(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    row_count: int,
    feature_count: int,
    matrix_name: str,
    scalar_specs: list[tuple[str, str, np.dtype[Any], tuple[int, ...]]],
    output_dir: Path,
    build_id: str,
    batch_size: int,
    created: list[Path],
) -> dict[str, dict[str, Any]]:
    matrix_path = output_dir / f"{matrix_name}-{build_id}.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(row_count, feature_count)
    )
    created.append(matrix_path)
    scalar_arrays: dict[str, np.memmap] = {}
    scalar_paths: dict[str, Path] = {}
    for _, artifact_name, dtype, shape in scalar_specs:
        path = output_dir / f"{artifact_name}-{build_id}.npy"
        scalar_arrays[artifact_name] = np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=shape
        )
        scalar_paths[artifact_name] = path
        created.append(path)
    result = connection.execute(query)
    reader = _arrow_record_batch_reader(result, batch_size)
    offset = 0
    scalar_columns = {source: artifact for source, artifact, _, _ in scalar_specs}
    for batch in reader:
        stop = offset + batch.num_rows
        if stop > row_count:
            raise GraphBuildError(f"{matrix_name} query returned too many rows")
        names = batch.schema.names
        for feature_index in range(feature_count):
            column = batch.column(names.index(f"feature_{feature_index:04d}"))
            matrix[offset:stop, feature_index] = np.asarray(
                column.to_numpy(zero_copy_only=False), dtype=np.float32
            )
        for source, artifact in scalar_columns.items():
            column = batch.column(names.index(source))
            scalar_arrays[artifact][offset:stop] = np.asarray(
                column.to_numpy(zero_copy_only=False), dtype=scalar_arrays[artifact].dtype
            )
        offset = stop
    if offset != row_count:
        raise GraphBuildError(f"{matrix_name} query returned {offset} rows, expected {row_count}")
    matrix.flush()
    for array in scalar_arrays.values():
        array.flush()
    output = {
        matrix_name: _array_metadata(matrix_path, matrix),
        **{
            name: _array_metadata(scalar_paths[name], array)
            for name, array in scalar_arrays.items()
        },
    }
    return output


def _stream_scalar_query(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    row_count: int,
    specs: list[tuple[str, str, np.dtype[Any], tuple[int, ...]]],
    output_dir: Path,
    build_id: str,
    batch_size: int,
    created: list[Path],
) -> dict[str, dict[str, Any]]:
    arrays: dict[str, np.memmap] = {}
    paths: dict[str, Path] = {}
    for _, artifact, dtype, shape in specs:
        path = output_dir / f"{artifact}-{build_id}.npy"
        arrays[artifact] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        paths[artifact] = path
        created.append(path)
    reader = _arrow_record_batch_reader(connection.execute(query), batch_size)
    offset = 0
    sources = {source: artifact for source, artifact, _, _ in specs}
    for batch in reader:
        stop = offset + batch.num_rows
        if stop > row_count:
            raise GraphBuildError("streamed query returned too many rows")
        for source, artifact in sources.items():
            arrays[artifact][offset:stop] = np.asarray(
                batch.column(batch.schema.names.index(source)).to_numpy(zero_copy_only=False),
                dtype=arrays[artifact].dtype,
            )
        offset = stop
    if offset != row_count:
        raise GraphBuildError(f"streamed query returned {offset} rows, expected {row_count}")
    for array in arrays.values():
        array.flush()
    return {name: _array_metadata(paths[name], array) for name, array in arrays.items()}


def _array_metadata(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
    }


def _manifest_artifacts(manifest_path: Path) -> set[Path]:
    """Return only prior files explicitly referenced by our own array manifest."""
    if not manifest_path.is_file():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if manifest.get("format") != MEMMAP_GRAPH_FORMAT:
        return set()
    root = manifest_path.parent.resolve()
    raw_arrays = manifest.get("arrays")
    if not isinstance(raw_arrays, dict):
        return set()
    values = [
        *(item.get("path") for item in raw_arrays.values() if isinstance(item, dict)),
        manifest.get("drug_metadata_path"),
    ]
    artifacts: set[Path] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        candidate = (root / value).resolve()
        if root in candidate.parents:
            artifacts.add(candidate)
    return artifacts


def _write_graph_descriptor(
    graph_path: Path, *, array_manifest_path: Path, torch: Any
) -> None:
    relative_manifest = os.path.relpath(array_manifest_path, graph_path.parent).replace("\\", "/")
    descriptor = {
        "format": MEMMAP_GRAPH_FORMAT,
        "format_version": MEMMAP_GRAPH_VERSION,
        "manifest_path": relative_manifest,
    }
    temporary = graph_path.with_suffix(".pt.tmp")
    torch.save(descriptor, temporary)
    os.replace(temporary, graph_path)


def _write_legacy_graph(
    graph_path: Path,
    *,
    bundle: GraphArrayBundle,
    auxiliary_targets: tuple[str, ...],
    torch: Any,
) -> None:
    """Explicit compatibility path; intentionally materializes PyG edge copies."""
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - checked by graph extra
        raise GraphBuildError("legacy graph storage requires torch-geometric") from exc
    arrays = bundle.arrays
    split_ids = np.asarray(arrays["patient_split_id"])
    train_mask = split_ids == 0
    validation_mask = split_ids == 1
    test_mask = split_ids == 2
    patient_ids = np.asarray(arrays["edge_patient_index"])
    drug_ids = np.asarray(arrays["edge_drug_index"])
    edge_train_mask = train_mask[patient_ids]
    edge_validation_mask = validation_mask[patient_ids]
    edge_test_mask = test_mask[patient_ids]
    graph = HeteroData()
    graph["patient"].x = torch.from_numpy(np.asarray(arrays["patient_x"]))
    graph["patient"].y = torch.from_numpy(np.asarray(arrays["patient_y"])).long()
    graph["patient"].train_mask = torch.from_numpy(train_mask)
    graph["patient"].val_mask = torch.from_numpy(validation_mask)
    graph["patient"].test_mask = torch.from_numpy(test_mask)
    graph["patient"].split_id = torch.from_numpy(split_ids)
    graph["patient"].primaryid = torch.from_numpy(
        np.asarray(arrays["patient_primaryid"])
    )
    graph["patient"].feature_names = list(bundle.manifest["patient_feature_names"])
    for target in auxiliary_targets:
        graph["patient"][target] = torch.from_numpy(
            np.asarray(arrays[f"patient_{target}"])
        ).long()
    graph["drug"].x = torch.from_numpy(np.asarray(arrays["drug_x"]))
    graph["drug"].node_id = torch.from_numpy(np.asarray(arrays["drug_node_id"])).long()
    node_metadata = pq.read_table(
        bundle.manifest_path.parent / bundle.manifest["drug_metadata_path"]
    )
    graph["drug"].node_label = node_metadata["node_label"].to_pylist()
    reverse = torch.stack(
        (torch.from_numpy(drug_ids).long(), torch.from_numpy(patient_ids).long())
    )
    forward = torch.stack(
        (
            torch.from_numpy(patient_ids[edge_train_mask]).long(),
            torch.from_numpy(drug_ids[edge_train_mask]).long(),
        )
    )
    graph[("patient", "takes", "drug")].edge_index = forward
    reverse_store = graph[("drug", "taken_by", "patient")]
    reverse_store.edge_index = reverse
    reverse_store.train_mask = torch.from_numpy(edge_train_mask)
    reverse_store.val_mask = torch.from_numpy(edge_validation_mask)
    reverse_store.test_mask = torch.from_numpy(edge_test_mask)
    graph.inductive_protocol = bundle.manifest["inductive_protocol"]
    temporary = graph_path.with_suffix(".pt.tmp")
    torch.save(graph, temporary)
    os.replace(temporary, graph_path)


def _quantile_matrix_for_split(
    xgb: Any,
    features: np.ndarray,
    labels: np.ndarray,
    split_ids: np.ndarray,
    *,
    split_id: int,
    batch_size: int,
    reference: Any | None = None,
) -> Any:
    """Build a histogram matrix from bounded memmap slices."""

    class SplitDataIter(xgb.DataIter):
        def __init__(self) -> None:
            super().__init__(release_data=True)
            self._offset = 0

        def reset(self) -> None:
            self._offset = 0

        def next(self, input_data: Any) -> int:
            while self._offset < features.shape[0]:
                start = self._offset
                stop = min(start + batch_size, features.shape[0])
                self._offset = stop
                selected = np.asarray(split_ids[start:stop]) == split_id
                if not selected.any():
                    continue
                input_data(
                    data=np.asarray(features[start:stop][selected], dtype=np.float32),
                    label=np.asarray(labels[start:stop][selected], dtype=np.float32),
                )
                return 1
            return 0

    return xgb.QuantileDMatrix(
        SplitDataIter(),
        max_bin=256,
        ref=reference,
        nthread=0,
    )


def _artifact_provenance(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    stat = source.stat()
    metadata: dict[str, Any] = {
        "path": str(source),
        "size_bytes": stat.st_size,
        "modified_at_utc": _utc_from_timestamp(stat.st_mtime),
    }
    with suppress(OSError, ValueError):
        metadata["parquet_rows"] = pq.ParquetFile(source).metadata.num_rows
    return metadata


def _utc_from_timestamp(value: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(tz=UTC).isoformat()


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _prepare_graph_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    paths: dict[str, Path],
    top_unknown: int,
    extra_patient_columns: tuple[str, ...] = (),
) -> None:
    cohort = _sql_literal(paths["cohort"].as_posix())
    splits = _sql_literal(paths["splits"].as_posix())
    dictionary = _sql_literal(paths["dictionary"].as_posix())
    edges = _sql_literal(paths["edges"].as_posix())
    extra_select = "".join(f", c.{name}" for name in extra_patient_columns)
    connection.execute(
        f"""
        CREATE TABLE patients AS
        SELECT row_number() OVER (ORDER BY c.report_date, c.primaryid) - 1 AS patient_index,
               c.primaryid, c.caseid, c.age, c.sex, c.num_drugs, c.is_serious, s.split
               {extra_select}
        FROM read_parquet('{cohort}') c
        JOIN read_parquet('{splits}') s USING (caseid)
        """
    )
    cohort_rows = connection.execute(f"SELECT count(*) FROM read_parquet('{cohort}')").fetchone()[0]
    patient_rows = connection.execute("SELECT count(*) FROM patients").fetchone()[0]
    if cohort_rows != patient_rows:
        raise GraphBuildError("case_splits join changed the cohort row count")
    mismatch = connection.execute(
        f"""
        SELECT count(*) FROM read_parquet('{cohort}') c
        JOIN read_parquet('{splits}') s USING (caseid)
        WHERE c.split <> s.split
        """
    ).fetchone()[0]
    if mismatch:
        raise GraphBuildError(f"cohort/case_splits mismatch for {mismatch} patients")

    connection.execute(
        f"""
        CREATE TABLE raw_frequency AS
        SELECT trim(e.drugname) AS faers_raw,
               count(*) FILTER (WHERE p.split = 'train')::BIGINT AS train_frequency,
               count(*)::BIGINT AS all_frequency
        FROM read_parquet('{edges}') e
        JOIN patients p USING (primaryid)
        WHERE e.drugname IS NOT NULL AND trim(e.drugname) <> ''
          AND NOT EXISTS (
              SELECT 1 FROM read_parquet('{dictionary}') d
              WHERE d.faers_raw = trim(e.drugname) AND d.dc_id > 0
          )
        GROUP BY trim(e.drugname)
        """
    )
    connection.execute(
        f"""
        CREATE TABLE top_unknown AS
        SELECT faers_raw, train_frequency,
               row_number() OVER (ORDER BY train_frequency DESC, faers_raw) AS unknown_rank
        FROM raw_frequency
        WHERE train_frequency > 0
        ORDER BY train_frequency DESC, faers_raw
        LIMIT {int(top_unknown)}
        """
    )
    max_dc_id = connection.execute(
        f"SELECT coalesce(max(dc_id), 0) FROM read_parquet('{dictionary}') WHERE dc_id > 0"
    ).fetchone()[0]
    connection.execute(
        f"""
        CREATE TABLE mapped_report_dc AS
        SELECT DISTINCT e.primaryid, d.dc_id::BIGINT AS dc_id
        FROM read_parquet('{edges}') e
        JOIN patients p USING (primaryid)
        JOIN read_parquet('{dictionary}') d ON d.faers_raw = trim(e.drugname)
        WHERE d.dc_id > 0
        """
    )
    connection.execute(
        """
        CREATE TABLE train_ror AS
        WITH totals AS (
            SELECT count(*) FILTER (WHERE is_serious = 1) AS total_serious,
                   count(*) FILTER (WHERE is_serious = 0) AS total_nonserious
            FROM patients WHERE split = 'train'
        ), counts AS (
            SELECT e.dc_id,
                   count(*) FILTER (WHERE p.is_serious = 1) AS a,
                   count(*) FILTER (WHERE p.is_serious = 0) AS b
            FROM mapped_report_dc e
            JOIN patients p USING (primaryid)
            WHERE p.split = 'train'
            GROUP BY e.dc_id
        ), cells AS (
            SELECT dc_id, a, b, total_serious - a AS c, total_nonserious - b AS d
            FROM counts CROSS JOIN totals
        )
        SELECT dc_id,
               CASE WHEN a = 0 OR b = 0 OR c = 0 OR d = 0
                    THEN ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
                    ELSE (a::DOUBLE * d::DOUBLE) / (b::DOUBLE * c::DOUBLE)
               END::DOUBLE AS ror
        FROM cells
        """
    )
    connection.execute(
        f"""
        CREATE TABLE known_nodes AS
        WITH metadata AS (
            SELECT dc_id, max(has_boxed_warning)::INTEGER AS has_boxed_warning,
                   any_value(atc_code) AS atc_code
            FROM read_parquet('{dictionary}') WHERE dc_id > 0 GROUP BY dc_id
        )
        SELECT DISTINCT e.dc_id::BIGINT AS semantic_id,
               'DC:' || e.dc_id::VARCHAR AS node_label,
               'mapped' AS node_kind,
               r.ror, m.has_boxed_warning, m.atc_code
        FROM mapped_report_dc e
        JOIN metadata m ON m.dc_id = e.dc_id
        LEFT JOIN train_ror r ON r.dc_id = e.dc_id
        """
    )
    has_other = connection.execute(
        """
        SELECT EXISTS(
            SELECT 1 FROM raw_frequency r
            WHERE NOT EXISTS (
                SELECT 1 FROM top_unknown t WHERE t.faers_raw = r.faers_raw
            )
        )
        """
    ).fetchone()[0]
    other_sql = (
        "SELECT -1::BIGINT, 'OTHER', 'other', NULL::DOUBLE, 0::INTEGER, NULL::VARCHAR"
        if has_other
        else "SELECT NULL::BIGINT, NULL, NULL, NULL::DOUBLE, NULL::INTEGER, NULL WHERE false"
    )
    connection.execute(
        f"""
        CREATE TABLE drug_nodes AS
        SELECT row_number() OVER (
                   ORDER BY CASE node_kind
                                WHEN 'mapped' THEN 0
                                WHEN 'frequent_unknown' THEN 1
                                ELSE 2
                            END,
                            semantic_id
               ) - 1 AS node_index,
               semantic_id, node_label, node_kind, ror, has_boxed_warning, atc_code
        FROM (
            SELECT * FROM known_nodes
            UNION ALL
            SELECT {int(max_dc_id)} + unknown_rank AS semantic_id,
                   faers_raw AS node_label, 'frequent_unknown' AS node_kind,
                   NULL::DOUBLE AS ror, 0::INTEGER AS has_boxed_warning,
                   NULL::VARCHAR AS atc_code
            FROM top_unknown
            UNION ALL
            {other_sql}
        ) nodes
        """
    )
    connection.execute(
        f"""
        CREATE TABLE graph_edges AS
        WITH known_edges AS (
            SELECT p.patient_index, e.dc_id::BIGINT AS semantic_id
            FROM mapped_report_dc e
            JOIN patients p USING (primaryid)
        ), unknown_edges AS (
            SELECT DISTINCT p.patient_index,
                   coalesce({int(max_dc_id)} + t.unknown_rank, -1)::BIGINT AS semantic_id
            FROM read_parquet('{edges}') e
            JOIN patients p USING (primaryid)
            LEFT JOIN top_unknown t ON t.faers_raw = trim(e.drugname)
            WHERE e.drugname IS NOT NULL AND trim(e.drugname) <> ''
              AND NOT EXISTS (
                  SELECT 1 FROM read_parquet('{dictionary}') d
                  WHERE d.faers_raw = trim(e.drugname) AND d.dc_id > 0
              )
        )
        SELECT DISTINCT links.patient_index, nodes.node_index
        FROM (SELECT * FROM known_edges UNION ALL SELECT * FROM unknown_edges) links
        JOIN drug_nodes nodes USING (semantic_id)
        """
    )
    orphan_patients = connection.execute(
        """
        SELECT count(*) FROM patients p
        WHERE NOT EXISTS (
            SELECT 1 FROM graph_edges e WHERE e.patient_index = p.patient_index
        )
        """
    ).fetchone()[0]
    if orphan_patients:
        raise GraphBuildError(f"graph contains {orphan_patients} patients without drug edges")


def _patient_features(
    ages: list[object],
    sexes: list[object],
    num_drugs: list[object],
    *,
    imputed_ages: list[object] | None = None,
    missing_age_fill: float = 0.0,
    enriched: dict[str, list[object]] | None = None,
    feature_order: tuple[str, ...] | None = None,
) -> np.ndarray:
    age_source = imputed_ages if imputed_ages is not None else ages
    age = np.asarray(
        [
            missing_age_fill
            if value is None or not math.isfinite(float(value))
            else float(value)
            for value in age_source
        ],
        dtype=np.float32,
    ) / 120.0
    sex = np.asarray([1.0 if str(value).strip().upper() == "M" else 0.0 for value in sexes])
    drugs = np.asarray([float(value or 0) for value in num_drugs], dtype=np.float32) / 50.0
    columns = [np.clip(age, 0, 1), sex, np.clip(drugs, 0, 1)]
    ordered_features = feature_order or (*ENRICHED_PATIENT_FEATURES, *PROSPECTIVE_PATIENT_FEATURES)
    for name in ordered_features:
        if enriched is not None and name in enriched:
            columns.append(
                np.asarray([float(value or 0) for value in enriched[name]], dtype=np.float32)
            )
    return np.column_stack(columns).astype(np.float32)


def _drug_features(table: object) -> np.ndarray:
    rows = table.to_pylist()
    mapped_logs = [math.log(max(float(row["ror"]), 1e-12)) for row in rows if row["ror"]]
    mean = float(np.mean(mapped_logs)) if mapped_logs else 0.0
    std = float(np.std(mapped_logs)) if mapped_logs and np.std(mapped_logs) > 0 else 1.0
    features = np.zeros((len(rows), 2 + len(ATC_INITIALS)), dtype=np.float32)
    index = {letter: offset for offset, letter in enumerate(ATC_INITIALS)}
    for row_number, row in enumerate(rows):
        if row["ror"] is not None:
            features[row_number, 0] = (math.log(max(float(row["ror"]), 1e-12)) - mean) / std
        features[row_number, 1] = float(row["has_boxed_warning"] or 0)
        for code in str(row["atc_code"] or "").split("|"):
            if code and code[0].upper() in index:
                features[row_number, 2 + index[code[0].upper()]] = 1.0
    return features


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute tie-aware ROC AUC without requiring scikit-learn."""
    y = np.asarray(y_true, dtype=np.int8)
    predictions = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise GraphBuildError("validation AUC requires both target classes")
    order = np.argsort(predictions, kind="mergesort")
    sorted_scores = predictions[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        end = start + 1
        while end < y.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[y == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _verify_masks(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> None:
    if not train.any() or not validation.any() or not test.any():
        raise GraphBuildError("train, validation, and test masks must all be non-empty")
    if np.any(train.astype(np.int8) + validation.astype(np.int8) + test.astype(np.int8) != 1):
        raise GraphBuildError("split masks must be mutually exclusive and exhaustive")


def _required_paths(data_dir: Path, *, cohort_path: Path | None = None) -> dict[str, Path]:
    paths = {
        "cohort": cohort_path or data_dir / "processed" / "tekarx_cohort.parquet",
        "splits": data_dir / "processed" / "case_splits.parquet",
        "dictionary": data_dir / "processed" / "drug_dictionary.parquet",
        "edges": data_dir / "processed" / "edges" / "report_drug.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise GraphBuildError(f"missing required graph inputs: {missing}")
    return paths


def _discover_patient_features(cohort_columns: set[str]) -> tuple[str, ...]:
    """Return an allow-listed, deterministic feature set without outcome leakage."""
    fixed = (
        *ENRICHED_PATIENT_FEATURES,
        *PROSPECTIVE_PATIENT_FEATURES,
        *NORMALIZED_DOSAGE_PATIENT_FEATURES,
    )
    features = [name for name in fixed if name in cohort_columns]
    # ATC class count names are data-driven, but their prefix is narrow enough to
    # avoid admitting labels or report outcomes into the model matrix.
    features.extend(
        sorted(
            name
            for name in cohort_columns
            if (
                (
                    name.startswith("atc_l1_count_")
                    and len(name.removeprefix("atc_l1_count_")) == 1
                )
                or (
                    name.startswith("indication_hash_")
                    and name.removeprefix("indication_hash_").isdigit()
                )
            )
            and name not in features
        )
    )
    return tuple(features)


def _write_manifest(
    path: Path,
    record: GraphBuildRecord,
    *,
    feature_names: tuple[str, ...],
    array_manifest: dict[str, Any],
) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "dataset": "TekaRx heterogeneous graph",
                "patient_features": list(feature_names),
                "intended_use": "research decision-support signal; not a diagnosis",
                "drug_features": list(DRUG_FEATURE_NAMES),
                "ror_scope": "train patients only; validation/test labels excluded",
                "unknown_vocabulary_scope": "train patients only",
                "message_passing": {
                    "patient_to_drug": "training edges only",
                    "drug_to_patient": "all edges with train/validation/test masks",
                    "held_out_patient_messages_to_shared_drugs": False,
                },
                "auxiliary_targets": {
                    "names": list(AUXILIARY_TARGETS),
                    "included_in_patient_x": False,
                },
                "storage": {
                    "format": record.storage_format,
                    "format_version": MEMMAP_GRAPH_VERSION,
                    "array_manifest_path": record.graph_array_manifest_path,
                    "arrays": array_manifest["arrays"],
                    "edge_order": array_manifest["edge_order"],
                    "materialization": array_manifest["materialization"],
                },
                "input_provenance": array_manifest["input_provenance"],
                "record": asdict(record),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")
