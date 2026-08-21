from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import HeteroData  # noqa: E402

from tekarx.modeling.gnn import (  # noqa: E402
    GNNTrainError,
    _select_feature_track,
    _validate_inductive_graph,
    aggregate_drug_neighbors,
    train_inductive_gnn,
)


def _tiny_graph() -> HeteroData:
    graph = HeteroData()
    graph["patient"].x = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.1, 1.0], [0.9, 0.0], [0.2, 1.0], [0.8, 0.0]]
    )
    graph["patient"].y = torch.tensor([0, 1, 0, 1, 0, 1])
    graph["patient"].train_mask = torch.tensor([True, True, False, False, False, False])
    graph["patient"].val_mask = torch.tensor([False, False, True, True, False, False])
    graph["patient"].test_mask = torch.tensor([False, False, False, False, True, True])
    graph["patient"].feature_names = ["feature_a", "feature_b"]
    graph["patient"].is_death = torch.tensor([0, 1, 0, 0, 0, 1])
    graph["drug"].x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    full_reverse = torch.tensor([[0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5]])
    graph[("drug", "taken_by", "patient")].edge_index = full_reverse
    graph[("drug", "taken_by", "patient")].train_mask = torch.tensor(
        [True, True, False, False, False, False]
    )
    graph[("drug", "taken_by", "patient")].val_mask = torch.tensor(
        [False, False, True, True, False, False]
    )
    graph[("drug", "taken_by", "patient")].test_mask = torch.tensor(
        [False, False, False, False, True, True]
    )
    graph[("patient", "takes", "drug")].edge_index = torch.tensor([[0, 1], [0, 1]])
    return graph


def test_neighbor_aggregation_is_per_patient_and_memory_bounded() -> None:
    result = aggregate_drug_neighbors(
        patient_count=3,
        drug_features=torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
        drug_to_patient_edge_index=torch.tensor([[0, 1, 0, 1], [0, 0, 1, 2]]),
        edge_chunk_size=1,
    )
    torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [2.0, 0.0], [0.0, 4.0]]))


def test_no_dosage_track_removes_only_reviewed_normalized_features() -> None:
    names = (
        "age_over_120",
        "exposure_dose_documented_fraction",
        "dose_normalized_relative_log_max",
        "dose_normalized_raw_edge_amount",
    )
    selected, selected_names = _select_feature_track(
        names, feature_track="prospective-no-dosage"
    )
    assert selected == [0, 1, 3]
    assert selected_names == (
        "age_over_120",
        "exposure_dose_documented_fraction",
        "dose_normalized_raw_edge_amount",
    )


def test_inductive_validation_rejects_heldout_patient_messages() -> None:
    graph = _tiny_graph()
    _validate_inductive_graph(graph, feature_names=("feature_a", "feature_b"))
    graph[("patient", "takes", "drug")].edge_index = torch.tensor([[0, 2], [0, 1]])
    with pytest.raises(GNNTrainError, match="held-out patients"):
        _validate_inductive_graph(graph, feature_names=("feature_a", "feature_b"))


def test_tiny_gnn_training_does_not_evaluate_test_by_default(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    graph_path = data_dir / "processed" / "tekarx_graph.pt"
    graph_path.parent.mkdir(parents=True)
    torch.save(_tiny_graph(), graph_path)

    record = train_inductive_gnn(
        data_dir=data_dir,
        epochs=2,
        batch_size=2,
        hidden_channels=4,
        patience=2,
        device="cpu",
        edge_chunk_size=2,
    )

    assert record.test_auc is None
    assert 0.0 <= record.validation_auc <= 1.0
    assert Path(record.model_path).is_file()
    checkpoint = torch.load(record.model_path, map_location="cpu", weights_only=False)
    assert checkpoint["test_auc"] is None
    assert "is_death" not in checkpoint["patient_feature_names"]
