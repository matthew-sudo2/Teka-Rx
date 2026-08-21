"""Estimate graph materialization memory before running a full FAERS build.

This is a deterministic capacity-planning model, not a process-RSS profiler.
It deliberately reports a lower bound for the legacy builder and the bounded
batch working set for the memory-mapped builder separately from on-disk bytes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphDimensions:
    """Cardinalities and dtypes that determine graph-array size."""

    patients: int
    drugs: int
    edges: int
    patient_features: int
    drug_features: int
    train_fraction: float = 0.70
    index_bytes: int = 4
    auxiliary_targets: int = 2

    def validate(self) -> None:
        values = (
            self.patients,
            self.drugs,
            self.edges,
            self.patient_features,
            self.drug_features,
        )
        if any(value < 1 for value in values):
            raise ValueError("graph dimensions must be positive")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between zero and one")
        if self.index_bytes not in (4, 8):
            raise ValueError("index_bytes must be 4 or 8")
        if self.auxiliary_targets < 0:
            raise ValueError("auxiliary_targets cannot be negative")


@dataclass(frozen=True)
class GraphMemoryEstimate:
    """Byte estimates for legacy and memory-mapped materialization."""

    sidecar_bytes: int
    legacy_minimum_peak_bytes: int
    mapped_batch_working_set_bytes: int
    materialization_batch_size: int

    @property
    def legacy_to_mapped_ratio(self) -> float:
        return self.legacy_minimum_peak_bytes / self.mapped_batch_working_set_bytes


def estimate_graph_memory(
    dimensions: GraphDimensions,
    *,
    materialization_batch_size: int = 131_072,
) -> GraphMemoryEstimate:
    """Return transparent byte estimates for graph materialization.

    The legacy lower bound counts the full patient matrix, three simultaneous
    full edge-index representations (query result, stacked index and reverse
    index), the train-only forward edge copy, and masks/labels. It excludes
    Python strings, Arrow validity buffers, allocator overhead, XGBoost and OS
    page cache, so real legacy peak RSS can be higher.

    The mapped working set counts one Arrow batch and its canonicalized output
    pages for the larger of patient or edge materialization. Sidecar files and
    OS page cache are reported separately; mapped file size is not equivalent
    to private resident RAM.
    """
    dimensions.validate()
    if materialization_batch_size < 1:
        raise ValueError("materialization_batch_size must be positive")

    patient_matrix = dimensions.patients * dimensions.patient_features * 4
    drug_matrix = dimensions.drugs * dimensions.drug_features * 4
    drug_node_ids = dimensions.drugs * 8
    one_edge_index = dimensions.edges * 2 * dimensions.index_bytes
    train_edge_index = round(one_edge_index * dimensions.train_fraction)
    # primaryid int64 + target int8 + split id int8 + one int8 per auxiliary target
    patient_metadata = dimensions.patients * (8 + 1 + 1 + dimensions.auxiliary_targets)
    sidecar_bytes = (
        patient_matrix
        + drug_matrix
        + drug_node_ids
        + one_edge_index
        + patient_metadata
    )

    legacy_minimum_peak = (
        patient_matrix
        + drug_matrix
        + drug_node_ids
        + 3 * one_edge_index
        + train_edge_index
        + patient_metadata
        + 3 * dimensions.edges
    )

    patient_batch_rows = min(dimensions.patients, materialization_batch_size)
    edge_batch_rows = min(dimensions.edges, materialization_batch_size)
    patient_batch = patient_batch_rows * dimensions.patient_features * 4 * 2
    edge_batch = edge_batch_rows * 2 * dimensions.index_bytes * 2
    mapped_working_set = max(patient_batch, edge_batch, drug_matrix)

    return GraphMemoryEstimate(
        sidecar_bytes=sidecar_bytes,
        legacy_minimum_peak_bytes=legacy_minimum_peak,
        mapped_batch_working_set_bytes=mapped_working_set,
        materialization_batch_size=materialization_batch_size,
    )


def _gibibytes(value: int) -> str:
    return f"{value / (1024**3):,.2f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate legacy versus memory-mapped graph materialization memory"
    )
    parser.add_argument("--patients", type=int, default=8_000_000)
    parser.add_argument("--drugs", type=int, default=4_000)
    parser.add_argument("--edges", type=int, default=24_000_000)
    parser.add_argument("--patient-features", type=int, default=118)
    parser.add_argument("--drug-features", type=int, default=28)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--index-bytes", type=int, choices=(4, 8), default=4)
    parser.add_argument("--auxiliary-targets", type=int, default=2)
    parser.add_argument("--materialization-batch-size", type=int, default=131_072)
    args = parser.parse_args()
    dimensions = GraphDimensions(
        patients=args.patients,
        drugs=args.drugs,
        edges=args.edges,
        patient_features=args.patient_features,
        drug_features=args.drug_features,
        train_fraction=args.train_fraction,
        index_bytes=args.index_bytes,
        auxiliary_targets=args.auxiliary_targets,
    )
    estimate = estimate_graph_memory(
        dimensions,
        materialization_batch_size=args.materialization_batch_size,
    )
    print("TekaRx graph materialization capacity estimate")
    print(f"  Memory-mapped sidecars on disk: {_gibibytes(estimate.sidecar_bytes)}")
    print(
        "  Pre-optimization legacy minimum arrays: "
        f"{_gibibytes(estimate.legacy_minimum_peak_bytes)}"
    )
    print(
        "  Memory-mapped batch working set: "
        f"{_gibibytes(estimate.mapped_batch_working_set_bytes)}"
    )
    print(f"  Estimated working-array reduction: {estimate.legacy_to_mapped_ratio:,.1f}x")
    print(
        "  Note: excludes XGBoost, Python/Arrow overhead and OS page cache; "
        "this is not measured peak RSS."
    )


if __name__ == "__main__":
    main()
