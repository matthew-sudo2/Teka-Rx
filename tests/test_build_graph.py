from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tekarx.transform.graph import (
    _discover_patient_features,
    _patient_features,
    _prepare_graph_tables,
    binary_auc,
    build_graph,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


def test_patient_features_and_tie_aware_auc() -> None:
    features = _patient_features([0, 60, 120], ["F", "M", None], [1, 5, 75])
    np.testing.assert_allclose(
        features,
        np.asarray([[0.0, 0.0, 0.02], [0.5, 1.0, 0.1], [1.0, 0.0, 1.0]], dtype=np.float32),
    )
    assert binary_auc(np.asarray([0, 1, 0, 1]), np.asarray([0.1, 0.8, 0.2, 0.8])) == 1.0

    enriched = _patient_features(
        [60],
        ["M"],
        [5],
        enriched={
            "max_ror": [3.0],
            "high_ror_count": [2],
            "has_boxed_warning": [1],
        },
    )
    assert enriched.shape == (1, 6)
    np.testing.assert_allclose(enriched[0, 3:], [3.0, 2.0, 1.0])

    imputed = _patient_features(
        [None, 90],
        [None, "F"],
        [1, 2],
        imputed_ages=[45, 90],
        enriched={"age_missing": [1, 0]},
    )
    np.testing.assert_allclose(imputed[:, 0], [0.375, 0.75])
    np.testing.assert_allclose(imputed[:, -1], [1.0, 0.0])


def test_feature_discovery_is_allow_listed_and_excludes_auxiliary_targets() -> None:
    discovered = _discover_patient_features(
        {
            "max_ror",
            "age_missing",
            "atc_l1_count_n",
            "indication_hash_03",
            "dose_normalized_relative_log_max",
            "dose_normalized_raw_edge_amount",
            "is_death",
            "is_hospitalization",
            "some_unreviewed_numeric_column",
        }
    )
    assert discovered == (
        "max_ror",
        "age_missing",
        "dose_normalized_relative_log_max",
        "atc_l1_count_n",
        "indication_hash_03",
    )
    matrix = _patient_features(
        [60],
        ["F"],
        [2],
        enriched={name: [offset] for offset, name in enumerate(discovered, start=1)},
        feature_order=discovered,
    )
    assert matrix.shape == (1, 3 + len(discovered))
    np.testing.assert_allclose(matrix[0, 3:], [1, 2, 3, 4, 5])


def test_graph_sql_keeps_top_unknown_and_collapses_rare_names(tmp_path: Path) -> None:
    data = tmp_path / "data" / "processed"
    cohort = data / "tekarx_cohort.parquet"
    splits = data / "case_splits.parquet"
    dictionary = data / "drug_dictionary.parquet"
    edges = data / "edges" / "report_drug.parquet"
    cohort_rows = [
        {
            "primaryid": str(index),
            "caseid": f"c{index}",
            "report_date": date(2024, 1, index),
            "age": 20 + index,
            "sex": "M" if index == 1 else "F",
            "num_drugs": 1,
            "is_serious": index % 2,
            "split": split,
        }
        for index, split in enumerate(("train", "validation", "test"), start=1)
    ]
    _write(cohort, cohort_rows)
    _write(
        splits,
        [
            {"caseid": row["caseid"], "max_report_date": row["report_date"], "split": row["split"]}
            for row in cohort_rows
        ],
    )
    _write(
        dictionary,
        [
            {
                "faers_raw": "ASPIRIN",
                "dc_id": 1,
                "atc_code": "B01AC06",
                "ror": 2.0,
                "has_boxed_warning": 1,
            }
        ],
    )
    _write(
        edges,
        [
            {"primaryid": "1", "drugname": "ASPIRIN"},
            {"primaryid": "1", "drugname": "UNKNOWN_TRAIN"},
            {"primaryid": "2", "drugname": "UNKNOWN_A"},
            {"primaryid": "2", "drugname": "UNKNOWN_A"},
            {"primaryid": "3", "drugname": "UNKNOWN_B"},
            {"primaryid": "3", "drugname": "UNKNOWN_C"},
        ],
    )
    connection = duckdb.connect()
    try:
        _prepare_graph_tables(
            connection,
            paths={
                "cohort": cohort,
                "splits": splits,
                "dictionary": dictionary,
                "edges": edges,
            },
            top_unknown=1,
        )
        nodes = connection.execute(
            "SELECT semantic_id, node_label, node_kind FROM drug_nodes ORDER BY node_index"
        ).fetchall()
        graph_edges = connection.execute(
            "SELECT patient_index, node_index FROM graph_edges ORDER BY ALL"
        ).fetchall()
    finally:
        connection.close()

    assert nodes == [
        (1, "DC:1", "mapped"),
        (2, "UNKNOWN_TRAIN", "frequent_unknown"),
        (-1, "OTHER", "other"),
    ]
    assert graph_edges == [(0, 0), (0, 1), (1, 2), (2, 2)]


def test_build_graph_saves_train_only_patient_to_drug_topology(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("xgboost")
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    (data_dir / "interim").mkdir(parents=True)
    cohort_rows = [
        {
            "primaryid": str(index),
            "caseid": f"c{index}",
            "report_date": date(2023 if split == "train" else 2024, 1, index),
            "age": None if index == 1 else 20 + index,
            "sex": "M" if index % 2 else "F",
            "num_drugs": 1,
            "is_serious": index % 2,
            "is_death": int(index == 1),
            "split": split,
        }
        for index, split in enumerate(
            ("train", "train", "validation", "validation", "test", "test"), start=1
        )
    ]
    _write(processed / "tekarx_cohort.parquet", cohort_rows)
    _write(
        processed / "case_splits.parquet",
        [
            {
                "caseid": row["caseid"],
                "max_report_date": row["report_date"],
                "split": row["split"],
            }
            for row in cohort_rows
        ],
    )
    _write(
        processed / "drug_dictionary.parquet",
        [
            {
                "faers_raw": "ASPIRIN",
                "dc_id": 1,
                "atc_code": "B01AC06",
                "ror": 2.0,
                "has_boxed_warning": 1,
            }
        ],
    )
    _write(
        processed / "edges" / "report_drug.parquet",
        [
            {"primaryid": str(index), "drugname": name}
            for index, name in enumerate(
                (
                    "ASPIRIN",
                    "TRAIN_UNKNOWN",
                    "FUTURE_UNKNOWN",
                    "ASPIRIN",
                    "ASPIRIN",
                    "FUTURE_UNKNOWN",
                ),
                start=1,
            )
        ],
    )

    record = build_graph(
        data_dir=data_dir,
        top_unknown=1,
        memory_limit="512MB",
        threads=1,
        xgb_rounds=1,
        xgb_early_stopping=1,
        storage_mode="legacy",
    )
    graph = torch.load(record.graph_path, map_location="cpu", weights_only=False)
    forward = graph[("patient", "takes", "drug")].edge_index
    reverse = graph[("drug", "taken_by", "patient")]

    assert bool(graph["patient"].train_mask[forward[0]].all())
    assert forward.shape[1] == int(reverse.train_mask.sum())
    assert int(reverse.val_mask.sum()) == 2
    assert int(reverse.test_mask.sum()) == 2
    assert graph["patient"].feature_names == [
        "age_over_120",
        "sex_binary",
        "num_drugs_over_50",
    ]
    assert float(graph["patient"].x[0, 0]) > 0.0
    assert "is_death" in graph["patient"]
