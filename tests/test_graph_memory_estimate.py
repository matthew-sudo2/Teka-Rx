import pytest

from benchmarks.graph_memory_estimate import GraphDimensions, estimate_graph_memory


def test_sidecar_estimate_matches_actual_array_contract() -> None:
    dimensions = GraphDimensions(
        patients=10,
        drugs=3,
        edges=20,
        patient_features=5,
        drug_features=2,
        index_bytes=4,
        auxiliary_targets=2,
    )
    estimate = estimate_graph_memory(dimensions, materialization_batch_size=4)
    patient_x = 10 * 5 * 4
    patient_scalars = 10 * (8 + 1 + 1 + 2)
    drug_x = 3 * 2 * 4
    drug_node_ids = 3 * 8
    canonical_edges = 20 * 2 * 4

    assert estimate.sidecar_bytes == (
        patient_x + patient_scalars + drug_x + drug_node_ids + canonical_edges
    )


def test_mapped_materialization_working_set_is_batch_bounded() -> None:
    small = estimate_graph_memory(
        GraphDimensions(
            patients=1_000_000,
            drugs=4_000,
            edges=3_000_000,
            patient_features=100,
            drug_features=28,
        ),
        materialization_batch_size=10_000,
    )
    large = estimate_graph_memory(
        GraphDimensions(
            patients=8_000_000,
            drugs=4_000,
            edges=24_000_000,
            patient_features=100,
            drug_features=28,
        ),
        materialization_batch_size=10_000,
    )

    assert small.mapped_batch_working_set_bytes == large.mapped_batch_working_set_bytes
    assert large.legacy_minimum_peak_bytes > small.legacy_minimum_peak_bytes * 7
    assert large.sidecar_bytes > small.sidecar_bytes * 7


def test_int32_indices_reduce_edge_storage_without_changing_patient_storage() -> None:
    base = dict(
        patients=8_000_000,
        drugs=4_000,
        edges=24_000_000,
        patient_features=118,
        drug_features=28,
    )
    int32 = estimate_graph_memory(GraphDimensions(**base, index_bytes=4))
    int64 = estimate_graph_memory(GraphDimensions(**base, index_bytes=8))

    expected_edge_difference = 24_000_000 * 2 * 4
    assert int64.sidecar_bytes - int32.sidecar_bytes == expected_edge_difference
    assert int32.legacy_minimum_peak_bytes < int64.legacy_minimum_peak_bytes


@pytest.mark.parametrize(
    "dimensions",
    [
        GraphDimensions(0, 1, 1, 1, 1),
        GraphDimensions(1, 1, 1, 1, 1, train_fraction=1.0),
        GraphDimensions(1, 1, 1, 1, 1, index_bytes=2),
    ],
)
def test_memory_estimate_rejects_invalid_dimensions(dimensions: GraphDimensions) -> None:
    with pytest.raises(ValueError):
        estimate_graph_memory(dimensions)
