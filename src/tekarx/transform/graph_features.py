"""Graph-derived features for TekaRx tabular models.

Computes static relational statistics from the patient-drug bipartite graph
WITHOUT training a GNN. Fully compliant with IMRAD rules.

All relational statistics are strictly computed on TRAINING EDGES ONLY to prevent
temporal leakage into validation or test sets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass
class GraphBundle:
    """Convenience container providing attribute and dict access to graph arrays."""

    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]

    def __getattr__(self, name: str) -> np.ndarray:
        if name in self.arrays:
            return self.arrays[name]
        raise AttributeError(f"'GraphBundle' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]


def load_graph_arrays(graph_dir: str | Path, *, mmap_mode: str | None = "r") -> GraphBundle:
    """Load graph arrays from descriptor file or directory containing manifest.json."""
    path = Path(graph_dir).resolve()
    if path.is_file() and path.suffix == ".pt":
        try:
            from tekarx.transform.graph_storage import load_graph_arrays as _load_storage

            storage_bundle = _load_storage(path, mmap_mode=mmap_mode)
            return GraphBundle(arrays=storage_bundle.arrays, manifest=storage_bundle.manifest)
        except Exception:
            pass

    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot find manifest.json at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_dir = manifest_path.parent
    arrays: dict[str, np.ndarray] = {}

    for name, meta in manifest.get("arrays", {}).items():
        arr_path = array_dir / meta["path"]
        if arr_path.is_file():
            arrays[name] = np.load(arr_path, mmap_mode=mmap_mode, allow_pickle=False)

    return GraphBundle(arrays=arrays, manifest=manifest)


def _load_train_edges(
    graph_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training edges and patient target/split arrays from graph directory."""
    bundle = load_graph_arrays(graph_dir)
    patient_split_id = bundle.patient_split_id
    patient_y = bundle.patient_y
    edge_patient = bundle.edge_patient_index
    edge_drug = bundle.edge_drug_index

    # Split id 0 corresponds to the train split
    train_patient_mask = patient_split_id == 0
    train_edge_mask = np.array(train_patient_mask[edge_patient])
    train_patient_idx = np.array(edge_patient[train_edge_mask])
    train_drug_idx = np.array(edge_drug[train_edge_mask])

    return train_patient_idx, train_drug_idx, patient_y, patient_split_id


def compute_drug_degree(
    edge_patient_index: np.ndarray,
    edge_drug_index: np.ndarray,
    n_drugs: int,
    n_patients: int | None = None,
) -> np.ndarray:
    """Count distinct patients per drug on training edges."""
    if n_patients is None:
        n_patients = int(edge_patient_index.max()) + 1 if len(edge_patient_index) > 0 else 0

    adj = sparse.csr_matrix(
        (np.ones(len(edge_patient_index), dtype=np.int8), (edge_patient_index, edge_drug_index)),
        shape=(n_patients, n_drugs),
    )
    return np.asarray(adj.sum(axis=0)).ravel().astype(np.float32)


def compute_drug_neighbor_ror(
    edge_patient_index: np.ndarray,
    edge_drug_index: np.ndarray,
    drug_x: np.ndarray,
    n_drugs: int | None = None,
    ror_idx: int = 0,
    n_patients: int | None = None,
) -> np.ndarray:
    """For each drug, calculate the average ROR of co-occurring neighbor drugs."""
    actual_n_drugs = drug_x.shape[0] if n_drugs is None else n_drugs
    if n_patients is None:
        n_patients = int(edge_patient_index.max()) + 1 if len(edge_patient_index) > 0 else 0

    adj = sparse.csr_matrix(
        (np.ones(len(edge_patient_index), dtype=np.int8), (edge_patient_index, edge_drug_index)),
        shape=(n_patients, actual_n_drugs),
    )
    cooccurrence = (adj.T @ adj).astype(np.float32)
    cooccurrence.setdiag(0.0)
    cooccurrence.eliminate_zeros()

    actual_ror_col = ror_idx if ror_idx < drug_x.shape[1] else 0
    raw_ror = drug_x[:, actual_ror_col].astype(np.float32)

    weighted_sum = cooccurrence @ raw_ror
    total_weight = np.asarray(cooccurrence.sum(axis=1)).ravel()

    neighbor_ror = np.zeros(actual_n_drugs, dtype=np.float32)
    nz = total_weight > 0
    neighbor_ror[nz] = weighted_sum[nz] / total_weight[nz]
    return neighbor_ror


def compute_cooccurrence_weight(
    edge_patient_index: np.ndarray,
    edge_drug_index: np.ndarray,
    n_drugs: int,
    n_patients: int | None = None,
) -> sparse.csr_matrix:
    """Count shared training patients for each drug pair."""
    if n_patients is None:
        n_patients = int(edge_patient_index.max()) + 1 if len(edge_patient_index) > 0 else 0

    adj = sparse.csr_matrix(
        (np.ones(len(edge_patient_index), dtype=np.int8), (edge_patient_index, edge_drug_index)),
        shape=(n_patients, n_drugs),
    )
    cooccurrence = (adj.T @ adj).astype(np.float32)
    cooccurrence.setdiag(0.0)
    cooccurrence.eliminate_zeros()
    return cooccurrence


def compute_propagated_risk(
    edge_patient_index: np.ndarray,
    edge_drug_index: np.ndarray,
    patient_y: np.ndarray,
    patient_split_id: np.ndarray | None = None,
    n_drugs: int | None = None,
    n_iter: int = 3,
    n_iterations: int | None = None,
    alpha: float = 0.7,
) -> np.ndarray:
    """Label propagation: each drug inherits mean seriousness of its training patients."""
    iterations = n_iterations if n_iterations is not None else n_iter

    # Resolve number of drugs
    if n_drugs is None:
        n_drugs = int(edge_drug_index.max()) + 1 if len(edge_drug_index) > 0 else 0

    n_patients = len(patient_y)
    if patient_split_id is not None:
        train_mask = patient_split_id == 0
        train_y = np.array(patient_y[train_mask]).astype(np.float32)
        orig_to_compact = -np.ones(n_patients, dtype=np.int64)
        compact_ids = np.arange(train_mask.sum())
        orig_to_compact[np.where(train_mask)[0]] = compact_ids
        compact_pt_idx = orig_to_compact[edge_patient_index]
        num_train_pts = len(compact_ids)
    else:
        train_y = patient_y.astype(np.float32)
        compact_pt_idx = edge_patient_index
        num_train_pts = n_patients

    adj = sparse.csr_matrix(
        (np.ones(len(compact_pt_idx), dtype=np.int8), (compact_pt_idx, edge_drug_index)),
        shape=(num_train_pts, n_drugs),
    )

    drug_counts = np.asarray(adj.sum(axis=0)).ravel()
    drug_serious = np.asarray(adj.T @ train_y).ravel()
    drug_risk = np.zeros(n_drugs, dtype=np.float32)
    nz = drug_counts > 0
    drug_risk[nz] = drug_serious[nz] / drug_counts[nz]

    cooccurrence = (adj.T @ adj).astype(np.float32)
    cooccurrence.setdiag(0.0)
    cooccurrence.eliminate_zeros()

    row_sums = np.asarray(cooccurrence.sum(axis=1)).ravel()
    inv_row_sums = np.zeros_like(row_sums)
    inv_row_sums[row_sums > 0] = 1.0 / row_sums[row_sums > 0]

    for _ in range(iterations):
        neighbor_avg = cooccurrence @ drug_risk * inv_row_sums
        drug_risk = alpha * drug_risk + (1.0 - alpha) * neighbor_avg

    return drug_risk.astype(np.float32)


def compute_cluster_risk(
    edge_patient_index: np.ndarray,
    edge_drug_index: np.ndarray,
    patient_y: np.ndarray,
    patient_split_id: np.ndarray | None = None,
    n_drugs: int | None = None,
) -> np.ndarray:
    """Community detection on drug co-occurrence graph -> assign risk per cluster."""
    if n_drugs is None:
        n_drugs = int(edge_drug_index.max()) + 1 if len(edge_drug_index) > 0 else 0

    n_patients = len(patient_y)
    if patient_split_id is not None:
        train_mask = patient_split_id == 0
        train_y = np.array(patient_y[train_mask]).astype(np.float32)
        orig_to_compact = -np.ones(n_patients, dtype=np.int64)
        compact_ids = np.arange(train_mask.sum())
        orig_to_compact[np.where(train_mask)[0]] = compact_ids
        compact_pt_idx = orig_to_compact[edge_patient_index]
        num_train_pts = len(compact_ids)
    else:
        train_y = patient_y.astype(np.float32)
        compact_pt_idx = edge_patient_index
        num_train_pts = n_patients

    adj = sparse.csr_matrix(
        (np.ones(len(compact_pt_idx), dtype=np.int8), (compact_pt_idx, edge_drug_index)),
        shape=(num_train_pts, n_drugs),
    )

    cooccurrence = (adj.T @ adj).astype(np.float32)
    cooccurrence.setdiag(0.0)
    cooccurrence.eliminate_zeros()

    n_components, labels = connected_components(cooccurrence, directed=False)
    cluster_risk_arr = np.zeros(n_components, dtype=np.float32)
    cluster_count_arr = np.zeros(n_components, dtype=np.float32)

    for d in range(n_drugs):
        cid = labels[d]
        pts = adj[:, d].nonzero()[0]
        if len(pts) == 0:
            continue
        cluster_risk_arr[cid] += float(train_y[pts].sum())
        cluster_count_arr[cid] += float(len(pts))

    cluster_risk_arr = np.divide(
        cluster_risk_arr,
        cluster_count_arr,
        out=np.zeros_like(cluster_risk_arr),
        where=cluster_count_arr > 0,
    )
    return cluster_risk_arr[labels]


def aggregate_to_patient(
    edge_patient_index: np.ndarray,
    edge_drug_index: np.ndarray,
    drug_values: np.ndarray,
    n_patients: int,
    n_drugs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a drug-level metric to patient level via mean and max pooling."""
    if n_drugs is None:
        n_drugs = len(drug_values)

    adj = sparse.csr_matrix(
        (np.ones(len(edge_patient_index), dtype=np.int8), (edge_patient_index, edge_drug_index)),
        shape=(n_patients, n_drugs),
    )

    means = np.asarray(adj @ drug_values).ravel()
    counts = np.asarray(adj.sum(axis=1)).ravel()
    safe_counts = np.where(counts > 0, counts, 1.0)
    means = (means / safe_counts).astype(np.float32)

    maxes = np.zeros(n_patients, dtype=np.float32)
    csr = adj.tocsr()
    for row in range(n_patients):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        if start == end:
            continue
        maxes[row] = float(drug_values[csr.indices[start:end]].max())

    return means, maxes


def build_graph_features(
    graph_dir: str | Path = "data/processed/tekarx_graph_arrays",
    output_path: str | Path | None = "data/processed/tekarx_graph_features.parquet",
    n_iterations: int = 3,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """Extract and aggregate all graph relational features for all patients."""
    bundle = load_graph_arrays(graph_dir)
    n_patients = bundle.patient_x.shape[0]
    n_drugs = bundle.drug_x.shape[0]
    drug_x = bundle.drug_x

    train_patient_idx, train_drug_idx, patient_y, patient_split_id = _load_train_edges(graph_dir)

    drug_degree = compute_drug_degree(train_patient_idx, train_drug_idx, n_drugs)
    drug_neighbor_ror = compute_drug_neighbor_ror(
        train_patient_idx, train_drug_idx, drug_x, n_drugs
    )
    drug_propagated_risk = compute_propagated_risk(
        train_patient_idx,
        train_drug_idx,
        patient_y,
        patient_split_id,
        n_drugs,
        n_iterations=n_iterations,
        alpha=alpha,
    )
    drug_cluster_risk = compute_cluster_risk(
        train_patient_idx,
        train_drug_idx,
        patient_y,
        patient_split_id,
        n_drugs,
    )

    # Full bipartite adjacency for patient aggregation
    full_adj = sparse.csr_matrix(
        (
            np.ones(len(bundle.edge_patient_index), dtype=np.int8),
            (bundle.edge_patient_index, bundle.edge_drug_index),
        ),
        shape=(n_patients, n_drugs),
    )

    def _agg(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.asarray(full_adj @ values).ravel()
        counts = np.asarray(full_adj.sum(axis=1)).ravel()
        safe = np.where(counts > 0, counts, 1.0)
        means = (means / safe).astype(np.float32)

        maxes = np.zeros(n_patients, dtype=np.float32)
        csr = full_adj.tocsr()
        for row in range(n_patients):
            s, e = csr.indptr[row], csr.indptr[row + 1]
            if s == e:
                continue
            maxes[row] = float(values[csr.indices[s:e]].max())
        return means, maxes

    avg_deg, max_deg = _agg(drug_degree)
    avg_ror, max_ror = _agg(drug_neighbor_ror)
    avg_prop, max_prop = _agg(drug_propagated_risk)
    avg_clust, max_clust = _agg(drug_cluster_risk)

    df = pd.DataFrame(
        {
            "primaryid": bundle.patient_primaryid.astype(str),
            "patient_avg_drug_degree": avg_deg,
            "patient_max_drug_degree": max_deg,
            "patient_avg_neighbor_ror": avg_ror,
            "patient_max_neighbor_ror": max_ror,
            "patient_avg_propagated_risk": avg_prop,
            "patient_max_propagated_risk": max_prop,
            "patient_avg_cluster_risk": avg_clust,
            "patient_max_cluster_risk": max_clust,
        }
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)

    return df