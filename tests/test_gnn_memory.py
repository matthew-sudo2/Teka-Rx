from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tekarx.modeling.gnn import (  # noqa: E402
    _aggregate_drug_neighbors_memmap,
    _chunked_train_standardization,
    _iter_split_batches,
    _load_training_graph,
    _normalized_matrix_batch,
    _training_label_counts,
    aggregate_drug_neighbors_from_arrays,
)


def _memmap(path: Path, values: np.ndarray) -> np.memmap:
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=values.dtype,
        shape=values.shape,
    )
    array[:] = values
    array.flush()
    return array


def test_memmap_neighbor_aggregation_matches_eager_reference(tmp_path: Path) -> None:
    drugs = _memmap(
        tmp_path / "drug_x.npy",
        np.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32),
    )
    edge_patient = _memmap(
        tmp_path / "edge_patient.npy",
        np.asarray([0, 0, 1, 2], dtype=np.int32),
    )
    edge_drug = _memmap(
        tmp_path / "edge_drug.npy",
        np.asarray([0, 1, 0, 1], dtype=np.int32),
    )

    mapped = _aggregate_drug_neighbors_memmap(
        patient_count=3,
        drug_features=drugs,
        edge_patient_index=edge_patient,
        edge_drug_index=edge_drug,
        output_path=tmp_path / "neighbors.npy",
        edge_chunk_size=1,
        torch=torch,
    )
    eager = aggregate_drug_neighbors_from_arrays(
        patient_count=3,
        drug_features=torch.from_numpy(np.asarray(drugs)),
        edge_patient_index=torch.from_numpy(np.asarray(edge_patient)),
        edge_drug_index=torch.from_numpy(np.asarray(edge_drug)),
        edge_chunk_size=1,
    )

    assert isinstance(mapped, np.memmap)
    np.testing.assert_allclose(mapped, eager.numpy())
    np.testing.assert_allclose(
        mapped,
        np.asarray([[1.0, 2.0], [2.0, 0.0], [0.0, 4.0]], dtype=np.float32),
    )


def test_chunked_statistics_match_eager_train_only_and_ignore_heldout_mutation(
    tmp_path: Path,
) -> None:
    values = _memmap(
        tmp_path / "patient_x.npy",
        np.asarray(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
                [6.0, 60.0, 600.0],
            ],
            dtype=np.float32,
        ),
    )
    split_id = _memmap(
        tmp_path / "split.npy",
        np.asarray([0, 0, 0, 1, 2, 2], dtype=np.int8),
    )
    expected = torch.from_numpy(np.asarray(values[:3, [0, 2]]).copy())
    expected_mean = expected.mean(dim=0)
    expected_std = expected.std(dim=0, unbiased=False)

    mean, std = _chunked_train_standardization(
        values,
        split_id,
        feature_indices=[0, 2],
        chunk_size=2,
        torch=torch,
    )
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(std, expected_std)

    values[3:] = 1_000_000.0
    values.flush()
    mutated_mean, mutated_std = _chunked_train_standardization(
        values,
        split_id,
        feature_indices=[0, 2],
        chunk_size=1,
        torch=torch,
    )
    torch.testing.assert_close(mutated_mean, mean)
    torch.testing.assert_close(mutated_std, std)


def test_minibatch_normalization_copies_only_selected_rows_and_columns(
    tmp_path: Path,
) -> None:
    original = np.arange(24, dtype=np.float32).reshape(6, 4)
    values = _memmap(tmp_path / "features.npy", original)
    ids = torch.tensor([1, 5])
    mean = torch.tensor([2.0, 4.0])
    std = torch.tensor([2.0, 4.0])

    batch = _normalized_matrix_batch(
        values,
        ids,
        feature_indices=[0, 2],
        mean=mean,
        std=std,
        torch=torch,
    )

    expected = (torch.from_numpy(original[[1, 5]][:, [0, 2]]) - mean) / std
    torch.testing.assert_close(batch, expected)
    np.testing.assert_array_equal(values, original)
    assert batch.shape == (2, 2)


def test_split_scanning_and_training_label_counts_never_select_heldout_rows(
    tmp_path: Path,
) -> None:
    split_id = _memmap(
        tmp_path / "split.npy",
        np.asarray([0, 1, 0, 2, 0, 2], dtype=np.int8),
    )
    labels = _memmap(
        tmp_path / "labels.npy",
        np.asarray([0, 9, 1, 9, 0, 9], dtype=np.int8),
    )

    batches = list(
        _iter_split_batches(
            split_id,
            split_value=0,
            row_count=6,
            scan_chunk_size=2,
            batch_size=1,
            shuffle=False,
            generator=None,
            torch=torch,
        )
    )
    selected = torch.cat(batches).tolist()
    assert selected == [0, 2, 4]
    assert _training_label_counts(labels, split_id, chunk_size=2, torch=torch) == (1, 2)


def test_memmap_trainer_architecture_avoids_full_standardized_feature_copy() -> None:
    loader_source = inspect.getsource(_load_training_graph)
    stats_source = inspect.getsource(_chunked_train_standardization)
    batch_source = inspect.getsource(_normalized_matrix_batch)
    aggregate_source = inspect.getsource(_aggregate_drug_neighbors_memmap)

    assert 'mmap_mode="r"' in loader_source
    assert "open_memmap" in aggregate_source
    assert "_iter_split_batches" in stats_source
    assert "values[train_mask]" not in stats_source
    assert "_gather_matrix" in batch_source
    assert ".sub_(mean).div_(std)" in batch_source
