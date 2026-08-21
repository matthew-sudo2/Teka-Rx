from __future__ import annotations

import inspect
import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tekarx.transform.graph import (
    _materialize_graph_arrays,
    _stream_feature_query,
    _stream_scalar_query,
    build_graph,
)
from tekarx.transform.graph_storage import MEMMAP_GRAPH_FORMAT, load_graph_arrays


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


def _write_graph_inputs(data_dir: Path) -> None:
    processed = data_dir / "processed"
    (data_dir / "interim").mkdir(parents=True)
    splits = ("train", "train", "validation", "validation", "test", "test")
    rows = [
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
        for index, split in enumerate(splits, start=1)
    ]
    _write(processed / "tekarx_cohort.parquet", rows)
    _write(
        processed / "case_splits.parquet",
        [
            {
                "caseid": row["caseid"],
                "max_report_date": row["report_date"],
                "split": row["split"],
            }
            for row in rows
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


def _build(data_dir: Path, *, storage_mode: str) -> object:
    _write_graph_inputs(data_dir)
    return build_graph(
        data_dir=data_dir,
        top_unknown=1,
        memory_limit="512MB",
        threads=1,
        xgb_rounds=1,
        xgb_early_stopping=1,
        storage_mode=storage_mode,
        materialization_batch_size=2,
        xgb_batch_size=2,
    )


def test_memory_mapped_graph_is_numerically_equivalent_to_legacy(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("xgboost")
    mapped_record = _build(tmp_path / "mapped" / "data", storage_mode="memory-mapped")
    legacy_record = _build(tmp_path / "legacy" / "data", storage_mode="legacy")

    bundle = load_graph_arrays(Path(mapped_record.graph_path), mmap_mode="r")
    legacy = torch.load(legacy_record.graph_path, map_location="cpu", weights_only=False)
    arrays = bundle.arrays

    assert mapped_record.storage_format == MEMMAP_GRAPH_FORMAT
    assert legacy_record.storage_format == "pyg"
    assert all(isinstance(array, np.memmap) for array in arrays.values())
    np.testing.assert_allclose(arrays["patient_x"], legacy["patient"].x.numpy())
    np.testing.assert_array_equal(arrays["patient_y"], legacy["patient"].y.numpy())
    np.testing.assert_allclose(arrays["drug_x"], legacy["drug"].x.numpy())
    np.testing.assert_array_equal(arrays["drug_node_id"], legacy["drug"].node_id.numpy())
    np.testing.assert_array_equal(
        arrays["patient_split_id"], legacy["patient"].split_id.numpy()
    )

    reverse = legacy[("drug", "taken_by", "patient")].edge_index.numpy()
    np.testing.assert_array_equal(arrays["edge_patient_index"], reverse[1])
    np.testing.assert_array_equal(arrays["edge_drug_index"], reverse[0])
    train_start, train_stop = bundle.manifest["edge_order"]["split_offsets"]["train"]
    expected_forward = np.stack(
        (
            arrays["edge_patient_index"][train_start:train_stop],
            arrays["edge_drug_index"][train_start:train_stop],
        )
    )
    np.testing.assert_array_equal(
        expected_forward,
        legacy[("patient", "takes", "drug")].edge_index.numpy(),
    )
    assert bundle.manifest["patient_feature_names"] == legacy["patient"].feature_names
    assert "is_death" not in bundle.manifest["patient_feature_names"]
    assert "patient_is_death" in arrays


def test_memmap_manifest_records_single_copy_and_inductive_split_semantics(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("xgboost")
    record = _build(tmp_path / "data", storage_mode="memory-mapped")
    graph_path = Path(record.graph_path)
    bundle = load_graph_arrays(graph_path, mmap_mode="r")
    manifest = bundle.manifest
    materialization = manifest["materialization"]

    descriptor = torch.load(graph_path, map_location="cpu", weights_only=True)
    assert descriptor["format"] == MEMMAP_GRAPH_FORMAT
    assert graph_path.stat().st_size < 64 * 1024
    assert materialization == {
        **materialization,
        "record_batch_rows": 2,
        "full_arrow_tables": False,
        "python_row_lists": False,
        "canonical_edge_copies": 1,
        "edge_split_string_array": False,
        "edge_boolean_masks_stored": False,
        "patient_boolean_masks_stored": False,
        "xgboost_input": "streamed QuantileDMatrix batches",
    }
    assert not any(name.endswith("mask") for name in manifest["arrays"])
    assert record.numeric_storage_bytes == sum(
        metadata["nbytes"] for metadata in manifest["arrays"].values()
    )
    for metadata in manifest["arrays"].values():
        path = bundle.manifest_path.parent / metadata["path"]
        assert path.is_file()
        assert path.stat().st_size >= metadata["nbytes"]

    patient_split = bundle.arrays["patient_split_id"]
    edge_patient = bundle.arrays["edge_patient_index"]
    offsets = manifest["edge_order"]["split_offsets"]
    for split_name, split_value in (("train", 0), ("validation", 1), ("test", 2)):
        start, stop = offsets[split_name]
        assert np.all(patient_split[edge_patient[start:stop]] == split_value)
    assert manifest["inductive_protocol"]["held_out_patient_messages_to_shared_drugs"] is False
    assert manifest["inductive_protocol"]["auxiliary_targets_in_x"] is False
    assert set(manifest["counts"]["edges_by_split"]) == {"train", "validation", "test"}
    assert json.loads(bundle.manifest_path.read_text(encoding="utf-8")) == manifest


def test_default_materializer_has_no_full_table_or_duplicate_edge_operations() -> None:
    production_source = "\n".join(
        inspect.getsource(function)
        for function in (
            _materialize_graph_arrays,
            _stream_feature_query,
            _stream_scalar_query,
        )
    )

    assert "to_arrow_reader" in production_source
    assert "open_memmap" in production_source
    assert ".to_pylist(" not in production_source
    assert ".to_arrow_table(" not in production_source
    assert ".fetchnumpy(" not in production_source
    assert "np.vstack(" not in production_source


@pytest.mark.filterwarnings("error:The given NumPy array is not writable")
def test_memory_mapped_descriptor_trains_without_scoring_test(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("xgboost")
    from tekarx.modeling.gnn import train_inductive_gnn

    data_dir = tmp_path / "data"
    graph_record = _build(data_dir, storage_mode="memory-mapped")
    train_record = train_inductive_gnn(
        data_dir=data_dir,
        graph_path=Path(graph_record.graph_path),
        epochs=2,
        batch_size=2,
        hidden_channels=4,
        patience=2,
        device="cpu",
        edge_chunk_size=1,
    )

    assert train_record.graph_storage == "numpy_memmap_v1"
    assert train_record.neighbor_storage == "temporary_numpy_memmap"
    assert train_record.test_auc is None
    checkpoint = torch.load(train_record.model_path, map_location="cpu", weights_only=False)
    assert checkpoint["test_auc"] is None
    manifest = json.loads(Path(train_record.manifest_path).read_text(encoding="utf-8"))
    assert manifest["leakage_controls"]["test_evaluated"] is False
    assert manifest["memory_strategy"] == {
        **manifest["memory_strategy"],
        "graph_storage": "numpy_memmap_v1",
        "neighbor_storage": "temporary_numpy_memmap",
        "patient_feature_selection": "columns gathered only for the active minibatch",
        "normalization_application": (
            "in place on copied minibatches; no cohort-sized normalized matrix"
        ),
        "edge_and_scan_chunk_size": 1,
    }
    assert not list(Path(graph_record.graph_path).parent.glob(".gnn-neighbors-*.npy"))
