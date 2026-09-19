"""Sharded full-graph visualization exporter.

Exports the complete patient-drug graph as size-bounded binary shards with a
multi-level LOD hierarchy, aggregate edges, and raw adjacency tiles.  The
browser template fetches shards on demand using frustum and zoom level.

This module extends the bounded-sample exporter in ``visualization.py``.
When ``--full`` is passed on the CLI the pipeline in this module runs instead
of the sample-based path.

Design principles (from ``docs/full_graph_visualization_plan.md``):
- Bounded working memory (depends on shard size, not total graph size)
- Progressive disclosure (density → clusters → individual patients)
- Immutable provenance
- No label leakage (outcome labels never influence layout)
- Determinism (same checkpoint + seed → identical output)
- Clinical restraint (research exploration tool, not diagnostic)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARD_FORMAT_VERSION = 1
EXPORTER_VERSION = "1.0.0"
DEFAULT_TARGET_SHARD_BYTES = 32 * 1024 * 1024  # 32 MiB
LOD_LEVELS = 4  # L0, L1, L2, L3
PATIENT_WRITE_CHUNK_SIZE = 1_000_000
EDGE_SCAN_CHUNK_SIZE = 1_000_000

SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}

REQUIRED_ARRAYS = (
    "patient_primaryid",
    "patient_y",
    "patient_split_id",
    "edge_patient_index",
    "edge_drug_index",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FullVisualizationError(RuntimeError):
    """Raised when the full-graph visualization pipeline fails."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardEntry:
    """One shard's metadata in the manifest."""

    path: str
    logical_type: str
    dtype: str
    byte_order: str
    shape: list[int]
    byte_length: int
    sha256: str
    index_range: list[int]
    spatial_bounds: list[float]
    lod_level: int
    source_checkpoint: str


@dataclass(frozen=True)
class LODCell:
    """One aggregate cell in the LOD hierarchy."""

    level: int
    cell_id: int
    patient_count: int
    serious_count: int
    nonserious_count: int
    split_counts: dict[str, int]
    edge_count: int
    dominant_drugs: list[int]
    dominant_atc: list[str]
    bbox: list[float]
    child_shard_ids: list[int]


@dataclass(frozen=True)
class FullVisualizationEstimate:
    """Preflight size estimate for a full-graph visualization export."""

    graph_patients: int
    graph_drugs: int
    graph_edges: int
    target_shard_bytes: int
    estimated_patient_coord_bytes: int
    estimated_patient_meta_bytes: int
    estimated_drug_bytes: int
    estimated_edge_bytes: int
    estimated_lod_bytes: int
    estimated_total_bytes: int
    estimated_shard_count: int
    estimated_cpu_working_set_bytes: int


@dataclass(frozen=True)
class FullVisualizationRecord:
    """Summary of a completed full-graph sharded visualization export."""

    output_path: str
    manifest_path: str
    provenance_path: str
    bundle_directory: str
    layout: str
    total_patients: int
    total_drugs: int
    total_edges: int
    patient_shard_count: int
    edge_shard_count: int
    lod_levels: int
    total_bytes: int
    seed: int
    elapsed_seconds: float
    source_checkpoint: str


@dataclass
class ExportProgress:
    """Mutable progress tracker for the export pipeline."""

    patients_written: int = 0
    edges_written: int = 0
    bytes_written: int = 0
    shards_written: int = 0
    start_time: float = field(default_factory=time.monotonic)

    def eta_seconds(self, total_items: int, items_done: int) -> float | None:
        elapsed = time.monotonic() - self.start_time
        if items_done == 0 or elapsed < 0.1:
            return None
        return elapsed * (total_items - items_done) / items_done


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def _read_graph_manifest(path: Path) -> dict[str, Any]:
    """Read and validate a graph array manifest."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullVisualizationError(f"cannot read graph manifest: {path}") from exc
    if manifest.get("format") != "tekarx.memmap_graph":
        raise FullVisualizationError(
            f"unsupported graph manifest format: {manifest.get('format')!r}"
        )
    return manifest


def _load_array(
    manifest_path: Path,
    manifest: dict[str, Any],
    name: str,
    *,
    required: bool = True,
) -> np.ndarray | None:
    """Memory-map one graph array from the manifest."""
    metadata = manifest.get("arrays", {}).get(name)
    if not isinstance(metadata, dict):
        if required:
            raise FullVisualizationError(f"graph manifest has no {name!r} array")
        return None
    relative = metadata.get("path")
    if not isinstance(relative, str):
        raise FullVisualizationError(f"graph array {name!r} has no path")
    root = manifest_path.parent.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        if not required:
            return None
        raise FullVisualizationError(f"missing graph array {name!r}: {path}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FullVisualizationError(f"cannot memory-map {name!r}: {path}") from exc
    if str(array.dtype) != metadata.get("dtype") or list(array.shape) != metadata.get("shape"):
        raise FullVisualizationError(f"array {name!r} does not match its manifest")
    return array


def _resolve_graph_manifest(data_dir: Path, graph_dir: Path | None) -> Path:
    """Find the graph array manifest, preferring explicit --graph-dir."""
    if graph_dir is not None:
        source = Path(graph_dir).resolve()
        candidates = (
            (source,)
            if source.is_file()
            else (
                source / "tekarx_graph_arrays" / "manifest.json",
                source / "manifest.json",
            )
        )
    else:
        checkpoints = data_dir / "processed" / "graph_visualization_checkpoints"
        candidates_list: list[Path] = []
        if checkpoints.is_dir():
            candidates_list.extend(
                directory / "tekarx_graph_arrays" / "manifest.json"
                for directory in sorted(checkpoints.iterdir(), reverse=True)
                if directory.is_dir() and not directory.name.startswith(".")
            )
        candidates_list.append(data_dir / "processed" / "tekarx_graph_arrays" / "manifest.json")
        candidates = tuple(candidates_list)
    for candidate in candidates:
        if candidate.is_file():
            try:
                m = _read_graph_manifest(candidate)
                for name in REQUIRED_ARRAYS:
                    meta = m.get("arrays", {}).get(name, {})
                    rel = meta.get("path")
                    if not isinstance(rel, str) or not (candidate.parent / rel).is_file():
                        break
                else:
                    return candidate.resolve()
            except FullVisualizationError:
                continue
    searched = ", ".join(str(p) for p in candidates)
    raise FullVisualizationError(f"no complete graph found; searched: {searched}")


# ---------------------------------------------------------------------------
# Provenance validation (§7)
# ---------------------------------------------------------------------------


def _validate_provenance(
    manifest_path: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> int:
    """Validate source arrays against the graph manifest.

    Returns the patient count.
    """
    patient_count = int(arrays["patient_y"].shape[0])
    if arrays["patient_primaryid"].shape != (patient_count,):
        raise FullVisualizationError("patient ID and target arrays have different lengths")
    if arrays["patient_split_id"].shape != (patient_count,):
        raise FullVisualizationError("patient split and target arrays have different lengths")
    if arrays["edge_patient_index"].shape != arrays["edge_drug_index"].shape:
        raise FullVisualizationError("edge patient and drug arrays have different lengths")

    # Verify dtypes match manifest
    for name, array in arrays.items():
        meta = manifest.get("arrays", {}).get(name, {})
        expected_dtype = meta.get("dtype")
        expected_shape = meta.get("shape")
        if expected_dtype and str(array.dtype) != expected_dtype:
            raise FullVisualizationError(
                f"array {name!r} dtype {array.dtype} != manifest {expected_dtype}"
            )
        if expected_shape and list(array.shape) != expected_shape:
            raise FullVisualizationError(
                f"array {name!r} shape {list(array.shape)} != manifest {expected_shape}"
            )
    return patient_count


# ---------------------------------------------------------------------------
# Size estimation (Phase A)
# ---------------------------------------------------------------------------


def estimate_full_visualization(
    *,
    data_dir: Path,
    graph_dir: Path | None = None,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
) -> FullVisualizationEstimate:
    """Estimate disk use, shard count, and memory budget without exporting.

    Reads only the graph manifest counts; does not touch the arrays themselves.
    """
    data_root = Path(data_dir).resolve()
    manifest_path = _resolve_graph_manifest(data_root, graph_dir)
    manifest = _read_graph_manifest(manifest_path)
    counts = manifest.get("counts", {})

    patients = int(counts.get("patient_nodes", 0))
    drugs = int(counts.get("drug_nodes", 0))
    edges = int(counts.get("patient_drug_edges", 0))

    if patients < 1 or drugs < 1:
        raise FullVisualizationError("graph manifest has no patients or drugs")

    # Patient coordinate shards: Float32 × 3 per patient
    patient_coord_bytes = patients * 3 * 4
    # Patient metadata shards: Uint8 y + Uint8 split = 2 bytes per patient
    patient_meta_bytes = patients * 2
    # Drug coordinates: Float32 × 3 per drug (single file)
    drug_bytes = drugs * 3 * 4 + 4096  # metadata overhead
    # Raw edge bytes: Int32 × 2 per edge
    edge_bytes = edges * 2 * 4
    # LOD overhead estimate: ~10% of patient coords + aggregate edges
    lod_bytes = max(patient_coord_bytes // 10, 1024 * 1024)

    total = patient_coord_bytes + patient_meta_bytes + drug_bytes + edge_bytes + lod_bytes

    # Shard count: patient coord shards + patient meta shards + edge shards
    patient_coord_shards = max(1, math.ceil(patient_coord_bytes / target_shard_bytes))
    patient_meta_shards = max(1, math.ceil(patient_meta_bytes / target_shard_bytes))
    edge_shards = max(1, math.ceil(edge_bytes / target_shard_bytes))
    shard_count = patient_coord_shards + patient_meta_shards + edge_shards + 4  # LOD files

    # CPU working set: one shard + edge scan chunk
    working_set = target_shard_bytes + EDGE_SCAN_CHUNK_SIZE * 2 * 4

    return FullVisualizationEstimate(
        graph_patients=patients,
        graph_drugs=drugs,
        graph_edges=edges,
        target_shard_bytes=target_shard_bytes,
        estimated_patient_coord_bytes=patient_coord_bytes,
        estimated_patient_meta_bytes=patient_meta_bytes,
        estimated_drug_bytes=drug_bytes,
        estimated_edge_bytes=edge_bytes,
        estimated_lod_bytes=lod_bytes,
        estimated_total_bytes=total,
        estimated_shard_count=shard_count,
        estimated_cpu_working_set_bytes=working_set,
    )


# ---------------------------------------------------------------------------
# Spatial bucket assignment (Phase B — §4.1)
# ---------------------------------------------------------------------------


def _compute_drug_degrees(
    edge_patients: np.ndarray,
    edge_drugs: np.ndarray,
    drug_count: int,
) -> np.ndarray:
    """Compute the degree (edge count) of each drug node."""
    degrees = np.zeros(drug_count, dtype=np.int64)
    for start in range(0, edge_patients.shape[0], EDGE_SCAN_CHUNK_SIZE):
        stop = min(start + EDGE_SCAN_CHUNK_SIZE, edge_patients.shape[0])
        chunk_drugs = np.asarray(edge_drugs[start:stop], dtype=np.int64)
        valid = (chunk_drugs >= 0) & (chunk_drugs < drug_count)
        degrees += np.bincount(chunk_drugs[valid], minlength=drug_count)
    return degrees


def _assign_patient_buckets(
    edge_patients: np.ndarray,
    edge_drugs: np.ndarray,
    patient_count: int,
    drug_count: int,
    drug_degrees: np.ndarray,
) -> np.ndarray:
    """Assign each patient to a primary-drug bucket.

    The primary drug is the highest-degree drug among the patient's edges.
    Ties are broken by drug index for determinism.  Patients with no edges
    are assigned to bucket ``drug_count`` (an overflow bucket).
    """
    # bucket[i] = primary drug index for patient i, or drug_count if none
    bucket = np.full(patient_count, drug_count, dtype=np.int32)
    best_degree = np.full(patient_count, -1, dtype=np.int64)

    for start in range(0, edge_patients.shape[0], EDGE_SCAN_CHUNK_SIZE):
        stop = min(start + EDGE_SCAN_CHUNK_SIZE, edge_patients.shape[0])
        patients_chunk = np.asarray(edge_patients[start:stop], dtype=np.int64)
        drugs_chunk = np.asarray(edge_drugs[start:stop], dtype=np.int64)
        valid = (
            (patients_chunk >= 0)
            & (patients_chunk < patient_count)
            & (drugs_chunk >= 0)
            & (drugs_chunk < drug_count)
        )
        p = patients_chunk[valid]
        d = drugs_chunk[valid]
        deg = drug_degrees[d]
        # Update where this drug's degree is higher, or same degree but lower index
        better = (deg > best_degree[p]) | ((deg == best_degree[p]) & (d < bucket[p]))
        update_patients = p[better]
        bucket[update_patients] = d[better].astype(np.int32)
        best_degree[update_patients] = deg[better]

    return bucket


def _hierarchical_bipartite_coordinates(
    patient_count: int,
    drug_count: int,
    patient_buckets: np.ndarray,
    patient_labels: np.ndarray,
    drug_degrees: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic bipartite layout coordinates.

    Patients are placed on the left (x ≈ −1.18), drugs on the right (x ≈ 1.18).
    Within the patient half, patients are grouped by their primary-drug bucket
    and placed deterministically using a seeded spatial hash.

    Outcome labels are used ONLY to add a small x-offset for visual
    separation; they do NOT influence bucket assignment or spatial position
    within the bucket.

    Returns (patient_coords [N×3 float32], drug_coords [M×3 float32]).
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    # --- Drug coordinates: stable ordering by degree then index ---

    drug_order = np.lexsort((np.arange(drug_count), -drug_degrees))
    drug_coords = np.zeros((drug_count, 3), dtype=np.float32)
    for rank, drug_idx in enumerate(drug_order):
        fraction = (rank + 0.5) / max(1, drug_count)
        angle = rank * golden_angle
        radius = 1.10 * np.sqrt(fraction)
        drug_coords[drug_idx, 0] = 1.18
        drug_coords[drug_idx, 1] = radius * np.cos(angle)
        drug_coords[drug_idx, 2] = radius * np.sin(angle)

    # --- Patient coordinates: grouped by bucket, seeded spatial hash ---
    patient_coords = np.zeros((patient_count, 3), dtype=np.float32)
    # Sort patients by bucket for locality
    order = np.argsort(patient_buckets)
    for rank, patient_idx in enumerate(order):
        fraction = (rank + 0.5) / max(1, patient_count)
        # Seeded spatial hash for deterministic placement
        hash_seed = (seed + int(patient_idx) * 104729) & 0xFFFFFFFF
        angle = (rank + (hash_seed % 104729)) * golden_angle
        radius = 1.48 * np.sqrt(fraction)
        label = float(np.asarray(patient_labels[patient_idx]))
        patient_coords[patient_idx, 0] = -1.18 + (label - 0.5) * 0.12
        patient_coords[patient_idx, 1] = radius * np.cos(angle)
        patient_coords[patient_idx, 2] = radius * np.sin(angle)

    return patient_coords, drug_coords


# ---------------------------------------------------------------------------
# Binary shard writing (Phase B)
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _write_shard(
    path: Path,
    data: np.ndarray,
    *,
    force: bool = False,
) -> tuple[int, str]:
    """Write one binary shard atomically.  Returns (byte_length, sha256).

    If a ``.done`` marker exists and ``force`` is False, skip writing.
    """
    marker = path.with_suffix(path.suffix + ".done")
    if not force and marker.is_file() and path.is_file():
        raw = path.read_bytes()
        return len(raw), _sha256_bytes(raw)

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data.tobytes()
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_bytes(raw)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

    # Atomic completion marker
    marker.write_bytes(b"")
    return len(raw), _sha256_bytes(raw)


def _patients_per_shard(patient_count: int, target_shard_bytes: int) -> int:
    """How many patients fit in one coordinate shard."""
    bytes_per_patient = 3 * 4  # Float32 × 3
    per_shard = max(1, target_shard_bytes // bytes_per_patient)
    return min(per_shard, patient_count)


def _write_patient_shards(
    bundle_dir: Path,
    patient_coords: np.ndarray,
    patient_labels: np.ndarray,
    patient_splits: np.ndarray,
    *,
    target_shard_bytes: int,
    source_checkpoint: str,
    resume: bool = False,
) -> tuple[list[ShardEntry], list[ShardEntry]]:
    """Write patient coordinate and metadata shards.

    Returns (coord_entries, meta_entries).
    """
    patient_count = patient_coords.shape[0]
    per_shard = _patients_per_shard(patient_count, target_shard_bytes)
    nodes_dir = bundle_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    coord_entries: list[ShardEntry] = []
    meta_entries: list[ShardEntry] = []

    shard_index = 0
    with tqdm(total=patient_count, desc="Writing patient shards", unit="patients") as pbar:
        for start in range(0, patient_count, per_shard):
            stop = min(start + per_shard, patient_count)
            chunk_size = stop - start

            # Coordinate shard
            coord_path = nodes_dir / f"patient_{shard_index:05d}.coords.bin"
            coord_data = np.ascontiguousarray(patient_coords[start:stop], dtype=np.float32)
            coord_bytes, coord_sha = _write_shard(
                coord_path, coord_data, force=not resume
            )

            bbox = [
                float(coord_data[:, 0].min()),
                float(coord_data[:, 1].min()),
                float(coord_data[:, 2].min()),
                float(coord_data[:, 0].max()),
                float(coord_data[:, 1].max()),
                float(coord_data[:, 2].max()),
            ]

            coord_entries.append(
                ShardEntry(
                    path=f"nodes/{coord_path.name}",
                    logical_type="patient_coordinates",
                    dtype="float32",
                    byte_order="little",
                    shape=[chunk_size, 3],
                    byte_length=coord_bytes,
                    sha256=coord_sha,
                    index_range=[start, stop],
                    spatial_bounds=bbox,
                    lod_level=3,
                    source_checkpoint=source_checkpoint,
                )
            )

            # Metadata shard (y + split packed together)
            meta_path = nodes_dir / f"patient_{shard_index:05d}.meta.bin"
            meta_y = np.asarray(patient_labels[start:stop], dtype=np.uint8)
            meta_split = np.asarray(patient_splits[start:stop], dtype=np.uint8)
            meta_data = np.column_stack([meta_y, meta_split]).astype(np.uint8)
            meta_bytes, meta_sha = _write_shard(
                meta_path, meta_data, force=not resume
            )
            meta_entries.append(
                ShardEntry(
                    path=f"nodes/{meta_path.name}",
                    logical_type="patient_metadata",
                    dtype="uint8",
                    byte_order="little",
                    shape=[chunk_size, 2],
                    byte_length=meta_bytes,
                    sha256=meta_sha,
                    index_range=[start, stop],
                    spatial_bounds=bbox,
                    lod_level=3,
                    source_checkpoint=source_checkpoint,
                )
            )

            shard_index += 1
            pbar.update(chunk_size)

    return coord_entries, meta_entries


def _write_drug_assets(
    bundle_dir: Path,
    drug_coords: np.ndarray,
    drug_count: int,
    drug_degrees: np.ndarray,
    manifest_path: Path,
    graph_manifest: dict[str, Any],
    source_checkpoint: str,
) -> tuple[ShardEntry, str]:
    """Write drug coordinates and metadata.  Returns (coord_entry, meta_path)."""

    nodes_dir = bundle_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    # Drug coordinate file
    coord_path = nodes_dir / "drugs.coords.bin"
    coord_data = np.ascontiguousarray(drug_coords, dtype=np.float32)
    coord_bytes, coord_sha = _write_shard(coord_path, coord_data, force=True)

    bbox = [
        float(coord_data[:, 0].min()),
        float(coord_data[:, 1].min()),
        float(coord_data[:, 2].min()),
        float(coord_data[:, 0].max()),
        float(coord_data[:, 1].max()),
        float(coord_data[:, 2].max()),
    ]

    coord_entry = ShardEntry(
        path=f"nodes/{coord_path.name}",
        logical_type="drug_coordinates",
        dtype="float32",
        byte_order="little",
        shape=[drug_count, 3],
        byte_length=coord_bytes,
        sha256=coord_sha,
        index_range=[0, drug_count],
        spatial_bounds=bbox,
        lod_level=3,
        source_checkpoint=source_checkpoint,
    )

    # Drug metadata JSON
    drug_metadata = _read_drug_metadata_from_graph(manifest_path, graph_manifest)
    drug_x = _load_array(manifest_path, graph_manifest, "drug_x", required=False)
    meta_list: list[dict[str, Any]] = []
    for idx in range(drug_count):
        source = drug_metadata.get(idx, {})
        features = _describe_drug_features(drug_x, idx)
        meta_list.append(
            {
                "index": idx,
                "source_index": idx,
                "label": str(source.get("node_label", f"Drug {idx}")),
                "semantic_id": int(source.get("semantic_id", 0)),
                "kind": str(source.get("node_kind", "unknown")),
                "degree": int(drug_degrees[idx]),
                **features,
            }
        )
    meta_path = nodes_dir / "drugs.meta.json"
    _write_json_atomic(meta_path, meta_list, compact=True)

    return coord_entry, f"nodes/{meta_path.name}"


def _read_drug_metadata_from_graph(
    manifest_path: Path, manifest: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    """Read drug metadata from the graph manifest's Parquet file."""
    import pyarrow.parquet as pq

    relative = manifest.get("drug_metadata_path")
    if not isinstance(relative, str):
        return {}
    path = (manifest_path.parent / relative).resolve()
    if not path.is_file():
        return {}
    try:
        table = pq.read_table(
            path, columns=["node_index", "semantic_id", "node_label", "node_kind"]
        )
        return {int(row["node_index"]): row for row in table.to_pylist()}
    except Exception:
        return {}


def _describe_drug_features(drug_x: np.ndarray | None, index: int) -> dict[str, Any]:
    """Extract display features for a single drug."""
    if drug_x is None or index >= drug_x.shape[0] or drug_x.shape[1] < 2:
        return {"ror_z": None, "boxed_warning": False, "atc": "?"}
    values = np.asarray(drug_x[index])
    atc = "?"
    if values.shape[0] >= 28 and float(np.max(values[2:28])) > 0.5:
        atc = chr(ord("A") + int(np.argmax(values[2:28])))
    return {
        "ror_z": round(float(values[0]), 4),
        "boxed_warning": bool(values[1] >= 0.5),
        "atc": atc,
    }


# ---------------------------------------------------------------------------
# LOD hierarchy (Phase C — §4.2)
# ---------------------------------------------------------------------------


def _build_lod_cells(
    patient_coords: np.ndarray,
    patient_labels: np.ndarray,
    patient_splits: np.ndarray,
    patient_buckets: np.ndarray,
    drug_count: int,
    drug_degrees: np.ndarray,
    edge_patients: np.ndarray,
    edge_drugs: np.ndarray,
) -> list[LODCell]:
    """Build a 4-level LOD hierarchy from patient coordinates.

    L0: Global density cells (4×4 grid over patient bounding box)
    L1: Drug-cluster cells (one per unique primary-drug bucket, capped at 64)
    L2: Micro-clusters within each L1 cell (subdivide spatially)
    L3: Individual patients (not stored as cells; that's the raw shard data)

    Returns a list of LODCell objects for levels L0–L2.
    """
    patient_count = patient_coords.shape[0]
    if patient_count == 0:
        return []

    # Compute per-patient edge counts for the aggregate
    patient_edge_counts = np.zeros(patient_count, dtype=np.int64)
    for start in range(0, edge_patients.shape[0], EDGE_SCAN_CHUNK_SIZE):
        stop = min(start + EDGE_SCAN_CHUNK_SIZE, edge_patients.shape[0])
        chunk_p = np.asarray(edge_patients[start:stop], dtype=np.int64)
        valid = (chunk_p >= 0) & (chunk_p < patient_count)
        patient_edge_counts += np.bincount(chunk_p[valid], minlength=patient_count)

    cells: list[LODCell] = []
    cell_id = 0

    # --- L0: 4×4 spatial grid ---
    y_min, y_max = float(patient_coords[:, 1].min()), float(patient_coords[:, 1].max())
    z_min, z_max = float(patient_coords[:, 2].min()), float(patient_coords[:, 2].max())
    y_range = max(y_max - y_min, 1e-6)
    z_range = max(z_max - z_min, 1e-6)

    l0_grid = 4
    l1_child_ids: dict[int, list[int]] = {}  # l0_cell_id -> list of l1 cell ids

    for gy in range(l0_grid):
        for gz in range(l0_grid):
            yl = y_min + gy * y_range / l0_grid
            yh = y_min + (gy + 1) * y_range / l0_grid
            zl = z_min + gz * z_range / l0_grid
            zh = z_min + (gz + 1) * z_range / l0_grid
            # last cell extends to include boundary
            if gy == l0_grid - 1:
                yh = y_max + 1e-6
            if gz == l0_grid - 1:
                zh = z_max + 1e-6

            mask = (
                (patient_coords[:, 1] >= yl)
                & (patient_coords[:, 1] < yh)
                & (patient_coords[:, 2] >= zl)
                & (patient_coords[:, 2] < zh)
            )
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue

            labels_in = np.asarray(patient_labels)[mask]
            splits_in = np.asarray(patient_splits)[mask]
            edges_in = patient_edge_counts[mask]
            buckets_in = patient_buckets[mask]

            # Dominant drugs: top 3 most common buckets
            if count > 0:
                bucket_counts = np.bincount(
                    buckets_in[buckets_in < drug_count].astype(np.int64),
                    minlength=drug_count,
                )
                top_drugs = list(map(int, np.argsort(-bucket_counts)[:3]))
            else:
                top_drugs = []

            l0_id = cell_id
            cells.append(
                LODCell(
                    level=0,
                    cell_id=l0_id,
                    patient_count=count,
                    serious_count=int(np.count_nonzero(labels_in == 1)),
                    nonserious_count=int(np.count_nonzero(labels_in == 0)),
                    split_counts={
                        "train": int(np.count_nonzero(splits_in == 0)),
                        "validation": int(np.count_nonzero(splits_in == 1)),
                        "test": int(np.count_nonzero(splits_in == 2)),
                    },
                    edge_count=int(edges_in.sum()),
                    dominant_drugs=top_drugs,
                    dominant_atc=[],
                    bbox=[float(patient_coords[mask, 0].min()), yl, zl,
                          float(patient_coords[mask, 0].max()), yh, zh],
                    child_shard_ids=[],
                )
            )
            l1_child_ids[l0_id] = []
            cell_id += 1

    # --- L1: Drug-cluster cells (group patients by primary-drug bucket) ---
    unique_buckets = np.unique(patient_buckets)
    # Cap at 64 biggest clusters; merge the rest into "other"
    if len(unique_buckets) > 64:
        bucket_sizes = np.array(
            [int(np.count_nonzero(patient_buckets == b)) for b in unique_buckets]
        )
        keep = unique_buckets[np.argsort(-bucket_sizes)[:64]]
        unique_buckets = np.sort(keep)

    l2_parent: dict[int, int] = {}
    for bucket_val in unique_buckets:
        mask = patient_buckets == bucket_val
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        labels_in = np.asarray(patient_labels)[mask]
        splits_in = np.asarray(patient_splits)[mask]
        edges_in = patient_edge_counts[mask]

        l1_id = cell_id

        # Attach to parent L0 cell
        if count > 0:
            centroid_y = float(patient_coords[mask, 1].mean())
            centroid_z = float(patient_coords[mask, 2].mean())
            for l0_cell in cells:
                if l0_cell.level == 0:
                    bb = l0_cell.bbox
                    if bb[1] <= centroid_y < bb[4] and bb[2] <= centroid_z < bb[5]:
                        if l0_cell.cell_id in l1_child_ids:
                            l1_child_ids[l0_cell.cell_id].append(l1_id)
                        break

        cells.append(
            LODCell(
                level=1,
                cell_id=l1_id,
                patient_count=count,
                serious_count=int(np.count_nonzero(labels_in == 1)),
                nonserious_count=int(np.count_nonzero(labels_in == 0)),
                split_counts={
                    "train": int(np.count_nonzero(splits_in == 0)),
                    "validation": int(np.count_nonzero(splits_in == 1)),
                    "test": int(np.count_nonzero(splits_in == 2)),
                },
                edge_count=int(edges_in.sum()),
                dominant_drugs=[int(bucket_val)] if int(bucket_val) < drug_count else [],
                dominant_atc=[],
                bbox=[
                    float(patient_coords[mask, 0].min()),
                    float(patient_coords[mask, 1].min()),
                    float(patient_coords[mask, 2].min()),
                    float(patient_coords[mask, 0].max()),
                    float(patient_coords[mask, 1].max()),
                    float(patient_coords[mask, 2].max()),
                ],
                child_shard_ids=[],
            )
        )
        l2_parent[l1_id] = int(bucket_val)
        cell_id += 1

    # Update L0 child lists
    for i, cell in enumerate(cells):
        if cell.level == 0 and cell.cell_id in l1_child_ids:
            cells[i] = LODCell(
                level=cell.level,
                cell_id=cell.cell_id,
                patient_count=cell.patient_count,
                serious_count=cell.serious_count,
                nonserious_count=cell.nonserious_count,
                split_counts=cell.split_counts,
                edge_count=cell.edge_count,
                dominant_drugs=cell.dominant_drugs,
                dominant_atc=cell.dominant_atc,
                bbox=cell.bbox,
                child_shard_ids=l1_child_ids[cell.cell_id],
            )

    # --- L2: Micro-clusters (split each L1 into 2×2 spatial sub-cells) ---
    l2_children: dict[int, list[int]] = {}
    for l1_cell in [c for c in cells if c.level == 1]:
        bucket_val = l2_parent.get(l1_cell.cell_id)
        if bucket_val is None:
            continue
        mask = patient_buckets == bucket_val
        indices = np.flatnonzero(mask)
        if len(indices) < 4:
            # Too few patients for sub-cells; skip L2
            continue

        coords_in = patient_coords[mask]
        labels_in = np.asarray(patient_labels)[mask]
        splits_in = np.asarray(patient_splits)[mask]
        edges_in = patient_edge_counts[mask]

        y_mid = float(np.median(coords_in[:, 1]))
        z_mid = float(np.median(coords_in[:, 2]))

        l2_children[l1_cell.cell_id] = []
        for qy in range(2):
            for qz in range(2):
                qmask = np.ones(len(indices), dtype=bool)
                if qy == 0:
                    qmask &= coords_in[:, 1] < y_mid
                else:
                    qmask &= coords_in[:, 1] >= y_mid
                if qz == 0:
                    qmask &= coords_in[:, 2] < z_mid
                else:
                    qmask &= coords_in[:, 2] >= z_mid

                qcount = int(np.count_nonzero(qmask))
                if qcount == 0:
                    continue

                l2_id = cell_id
                l2_children[l1_cell.cell_id].append(l2_id)
                cells.append(
                    LODCell(
                        level=2,
                        cell_id=l2_id,
                        patient_count=qcount,
                        serious_count=int(np.count_nonzero(labels_in[qmask] == 1)),
                        nonserious_count=int(np.count_nonzero(labels_in[qmask] == 0)),
                        split_counts={
                            "train": int(np.count_nonzero(splits_in[qmask] == 0)),
                            "validation": int(np.count_nonzero(splits_in[qmask] == 1)),
                            "test": int(np.count_nonzero(splits_in[qmask] == 2)),
                        },
                        edge_count=int(edges_in[qmask].sum()),
                        dominant_drugs=l1_cell.dominant_drugs,
                        dominant_atc=[],
                        bbox=[
                            float(coords_in[qmask, 0].min()),
                            float(coords_in[qmask, 1].min()),
                            float(coords_in[qmask, 2].min()),
                            float(coords_in[qmask, 0].max()),
                            float(coords_in[qmask, 1].max()),
                            float(coords_in[qmask, 2].max()),
                        ],
                        child_shard_ids=[],
                    )
                )
                cell_id += 1

    # Update L1 child lists
    for i, cell in enumerate(cells):
        if cell.level == 1 and cell.cell_id in l2_children:
            cells[i] = LODCell(
                level=cell.level,
                cell_id=cell.cell_id,
                patient_count=cell.patient_count,
                serious_count=cell.serious_count,
                nonserious_count=cell.nonserious_count,
                split_counts=cell.split_counts,
                edge_count=cell.edge_count,
                dominant_drugs=cell.dominant_drugs,
                dominant_atc=cell.dominant_atc,
                bbox=cell.bbox,
                child_shard_ids=l2_children[cell.cell_id],
            )

    return cells


def _write_lod_files(
    bundle_dir: Path,
    cells: list[LODCell],
    source_checkpoint: str,
) -> list[ShardEntry]:
    """Write LOD hierarchy to binary + JSON files.

    Returns shard entries for the manifest.
    """
    lod_dir = bundle_dir / "lod"
    lod_dir.mkdir(parents=True, exist_ok=True)

    entries: list[ShardEntry] = []

    # L0 nodes: density cells as JSON (small enough)
    l0_cells = [c for c in cells if c.level == 0]
    l0_data = [asdict(c) for c in l0_cells]
    l0_path = lod_dir / "l0_nodes.json"
    raw_l0 = json.dumps(l0_data, separators=(",", ":")).encode("utf-8")
    l0_path.write_bytes(raw_l0)
    entries.append(
        ShardEntry(
            path=f"lod/{l0_path.name}",
            logical_type="lod_l0_nodes",
            dtype="json",
            byte_order="n/a",
            shape=[len(l0_cells)],
            byte_length=len(raw_l0),
            sha256=_sha256_bytes(raw_l0),
            index_range=[0, len(l0_cells)],
            spatial_bounds=[],
            lod_level=0,
            source_checkpoint=source_checkpoint,
        )
    )

    # L0 edges: aggregate edge summaries as JSON
    l0_edges_data = [
        {
            "cell_id": c.cell_id,
            "edge_count": c.edge_count,
            "dominant_drugs": c.dominant_drugs,
        }
        for c in l0_cells
    ]
    l0_edges_path = lod_dir / "l0_edges.json"
    raw_l0e = json.dumps(l0_edges_data, separators=(",", ":")).encode("utf-8")
    l0_edges_path.write_bytes(raw_l0e)
    entries.append(
        ShardEntry(
            path=f"lod/{l0_edges_path.name}",
            logical_type="lod_l0_edges",
            dtype="json",
            byte_order="n/a",
            shape=[len(l0_cells)],
            byte_length=len(raw_l0e),
            sha256=_sha256_bytes(raw_l0e),
            index_range=[0, len(l0_cells)],
            spatial_bounds=[],
            lod_level=0,
            source_checkpoint=source_checkpoint,
        )
    )

    # L1 index
    l1_cells = [c for c in cells if c.level == 1]
    l1_data = [asdict(c) for c in l1_cells]
    l1_path = lod_dir / "l1_index.json"
    raw_l1 = json.dumps(l1_data, separators=(",", ":")).encode("utf-8")
    l1_path.write_bytes(raw_l1)
    entries.append(
        ShardEntry(
            path=f"lod/{l1_path.name}",
            logical_type="lod_l1_index",
            dtype="json",
            byte_order="n/a",
            shape=[len(l1_cells)],
            byte_length=len(raw_l1),
            sha256=_sha256_bytes(raw_l1),
            index_range=[0, len(l1_cells)],
            spatial_bounds=[],
            lod_level=1,
            source_checkpoint=source_checkpoint,
        )
    )

    # L2 index
    l2_cells = [c for c in cells if c.level == 2]
    l2_data = [asdict(c) for c in l2_cells]
    l2_path = lod_dir / "l2_index.json"
    raw_l2 = json.dumps(l2_data, separators=(",", ":")).encode("utf-8")
    l2_path.write_bytes(raw_l2)
    entries.append(
        ShardEntry(
            path=f"lod/{l2_path.name}",
            logical_type="lod_l2_index",
            dtype="json",
            byte_order="n/a",
            shape=[len(l2_cells)],
            byte_length=len(raw_l2),
            sha256=_sha256_bytes(raw_l2),
            index_range=[0, len(l2_cells)],
            spatial_bounds=[],
            lod_level=2,
            source_checkpoint=source_checkpoint,
        )
    )

    return entries


def _generate_aggregate_edges(
    cells: list[LODCell],
    edge_patients: np.ndarray,
    edge_drugs: np.ndarray,
    patient_buckets: np.ndarray,
    patient_coords: np.ndarray,
    drug_count: int,
) -> list[dict[str, Any]]:
    """Generate aggregate cluster-to-drug edge summaries for L0–L1.

    Returns a list of aggregate edge records.
    """
    aggregates: list[dict[str, Any]] = []

    # For each L1 cell (drug-bucket cluster), count edges to each drug

    l1_cells = [c for c in cells if c.level == 1]
    for l1_cell in l1_cells:
        if not l1_cell.dominant_drugs:
            continue
        bucket_val = l1_cell.dominant_drugs[0]
        mask = patient_buckets == bucket_val
        patient_indices = set(map(int, np.flatnonzero(mask)))
        if not patient_indices:
            continue

        # Scan edges for this cluster
        drug_counts: dict[int, int] = {}
        for start in range(0, edge_patients.shape[0], EDGE_SCAN_CHUNK_SIZE):
            stop = min(start + EDGE_SCAN_CHUNK_SIZE, edge_patients.shape[0])
            p_chunk = np.asarray(edge_patients[start:stop], dtype=np.int64)
            d_chunk = np.asarray(edge_drugs[start:stop], dtype=np.int64)
            for i in range(len(p_chunk)):
                if int(p_chunk[i]) in patient_indices:
                    d = int(d_chunk[i])
                    drug_counts[d] = drug_counts.get(d, 0) + 1

        # Top 10 drugs by count
        top = sorted(drug_counts.items(), key=lambda x: -x[1])[:10]
        for drug_idx, count in top:
            aggregates.append(
                {
                    "source_cell": l1_cell.cell_id,
                    "source_level": 1,
                    "target_drug": drug_idx,
                    "edge_count": count,
                }
            )

    return aggregates


def _write_aggregate_edge_shards(
    bundle_dir: Path,
    aggregates: list[dict[str, Any]],
    source_checkpoint: str,
) -> list[ShardEntry]:
    """Write aggregate edge summaries as JSON shards."""
    edges_dir = bundle_dir / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)

    if not aggregates:
        return []

    path = edges_dir / "aggregate_00000.json"
    raw = json.dumps(aggregates, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return [
        ShardEntry(
            path=f"edges/{path.name}",
            logical_type="aggregate_edges",
            dtype="json",
            byte_order="n/a",
            shape=[len(aggregates)],
            byte_length=len(raw),
            sha256=_sha256_bytes(raw),
            index_range=[0, len(aggregates)],
            spatial_bounds=[],
            lod_level=1,
            source_checkpoint=source_checkpoint,
        )
    ]


# ---------------------------------------------------------------------------
# Raw adjacency streaming (Phase D — §4.4)
# ---------------------------------------------------------------------------


def _partition_raw_edges(
    edge_patients: np.ndarray,
    edge_drugs: np.ndarray,
    patient_count: int,
    patients_per_shard: int,
) -> dict[int, list[tuple[int, int]]]:
    """Partition raw edges by patient shard.

    Returns {shard_index: [(patient_index, drug_index), ...]}.
    """
    partitions: dict[int, list[tuple[int, int]]] = {}

    for start in range(0, edge_patients.shape[0], EDGE_SCAN_CHUNK_SIZE):
        stop = min(start + EDGE_SCAN_CHUNK_SIZE, edge_patients.shape[0])
        p_chunk = np.asarray(edge_patients[start:stop], dtype=np.int64)
        d_chunk = np.asarray(edge_drugs[start:stop], dtype=np.int64)

        valid = (p_chunk >= 0) & (p_chunk < patient_count)
        p_valid = p_chunk[valid]
        d_valid = d_chunk[valid]

        shard_ids = (p_valid // patients_per_shard).astype(np.int64)
        for i in range(len(p_valid)):
            sid = int(shard_ids[i])
            if sid not in partitions:
                partitions[sid] = []
            partitions[sid].append((int(p_valid[i]), int(d_valid[i])))

    return partitions


def _write_adjacency_shards(
    bundle_dir: Path,
    partitions: dict[int, list[tuple[int, int]]],
    source_checkpoint: str,
) -> tuple[list[ShardEntry], dict[int, str]]:
    """Write raw adjacency shards partitioned by patient tile.

    Returns (shard_entries, tile_to_shard_path).
    """
    edges_dir = bundle_dir / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)

    entries: list[ShardEntry] = []
    tile_index: dict[int, str] = {}

    for shard_id in sorted(partitions):
        pairs = partitions[shard_id]
        if not pairs:
            continue

        data = np.array(pairs, dtype=np.int32)
        path = edges_dir / f"adjacency_{shard_id:05d}.bin"
        nbytes, sha = _write_shard(path, data, force=True)

        rel_path = f"edges/{path.name}"
        tile_index[shard_id] = rel_path
        entries.append(
            ShardEntry(
                path=rel_path,
                logical_type="raw_adjacency",
                dtype="int32",
                byte_order="little",
                shape=[len(pairs), 2],
                byte_length=nbytes,
                sha256=sha,
                index_range=[
                    min(p for p, _ in pairs),
                    max(p for p, _ in pairs) + 1,
                ],
                spatial_bounds=[],
                lod_level=3,
                source_checkpoint=source_checkpoint,
            )
        )

    return entries, tile_index


def _write_tile_edge_index(
    bundle_dir: Path,
    tile_index: dict[int, str],
) -> None:
    """Write a compact tile → edge shard lookup as JSON."""
    edges_dir = bundle_dir / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)
    index_path = edges_dir / "tile_index.json"
    _write_json_atomic(
        index_path,
        {str(k): v for k, v in sorted(tile_index.items())},
        compact=True,
    )


# ---------------------------------------------------------------------------
# Main export pipeline
# ---------------------------------------------------------------------------


def export_full_visualization(
    *,
    data_dir: Path,
    graph_dir: Path | None = None,
    layout: str = "hierarchical-bipartite",
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    seed: int = 42,
    output: Path | None = None,
    resume: bool = False,
) -> FullVisualizationRecord:
    """Export the complete patient-drug graph as a sharded visualization bundle.

    This is the ``--full`` mode entry point.
    """
    wall_start = time.monotonic()

    if layout not in ("hierarchical-bipartite",):
        raise ValueError(f"unsupported full layout: {layout!r}")
    if target_shard_bytes < 1:
        raise ValueError("target_shard_bytes must be at least 1")

    data_root = Path(data_dir).resolve()
    manifest_path = _resolve_graph_manifest(data_root, graph_dir)
    manifest = _read_graph_manifest(manifest_path)
    source_checkpoint = str(manifest_path.parent.name)

    # Load arrays
    arrays = {
        name: _load_array(manifest_path, manifest, name) for name in REQUIRED_ARRAYS
    }
    patient_count = _validate_provenance(manifest_path, manifest, arrays)

    counts = manifest.get("counts", {})
    drug_count = int(counts.get("drug_nodes", int(np.max(arrays["edge_drug_index"])) + 1))
    edge_count = int(counts.get("patient_drug_edges", arrays["edge_patient_index"].shape[0]))

    if drug_count < 1:
        raise FullVisualizationError("graph has no drug nodes")

    # Output paths
    default_name = "full_graph_visualization"
    output_path = Path(output or data_root / "processed" / f"{default_name}.html")
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()
    bundle_dir = output_path.parent / f"{output_path.stem}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Phase B: Spatial bucketing and patient coordinates
    print("Full viz 1/6: computing drug degrees and spatial buckets", flush=True)
    drug_degrees = _compute_drug_degrees(
        arrays["edge_patient_index"], arrays["edge_drug_index"], drug_count
    )
    patient_buckets = _assign_patient_buckets(
        arrays["edge_patient_index"],
        arrays["edge_drug_index"],
        patient_count,
        drug_count,
        drug_degrees,
    )

    print("Full viz 2/6: generating bipartite layout coordinates", flush=True)
    patient_coords, drug_coords = _hierarchical_bipartite_coordinates(
        patient_count,
        drug_count,
        patient_buckets,
        arrays["patient_y"],
        drug_degrees,
        seed=seed,
    )

    print("Full viz 3/6: writing patient shards", flush=True)
    coord_entries, meta_entries = _write_patient_shards(
        bundle_dir,
        patient_coords,
        arrays["patient_y"],
        arrays["patient_split_id"],
        target_shard_bytes=target_shard_bytes,
        source_checkpoint=source_checkpoint,
        resume=resume,
    )

    print("Full viz 4/6: writing drug assets and LOD hierarchy", flush=True)
    drug_entry, drug_meta_path = _write_drug_assets(
        bundle_dir,
        drug_coords,
        drug_count,
        drug_degrees,
        manifest_path,
        manifest,
        source_checkpoint,
    )

    # Phase C: LOD hierarchy
    lod_cells = _build_lod_cells(
        patient_coords,
        arrays["patient_y"],
        arrays["patient_split_id"],
        patient_buckets,
        drug_count,
        drug_degrees,
        arrays["edge_patient_index"],
        arrays["edge_drug_index"],
    )
    lod_entries = _write_lod_files(bundle_dir, lod_cells, source_checkpoint)

    # Aggregate edges
    aggregates = _generate_aggregate_edges(
        lod_cells,
        arrays["edge_patient_index"],
        arrays["edge_drug_index"],
        patient_buckets,
        patient_coords,
        drug_count,
    )
    agg_edge_entries = _write_aggregate_edge_shards(bundle_dir, aggregates, source_checkpoint)

    # Phase D: Raw adjacency
    print("Full viz 5/6: partitioning raw edges by patient tile", flush=True)
    per_shard = _patients_per_shard(patient_count, target_shard_bytes)
    partitions = _partition_raw_edges(
        arrays["edge_patient_index"],
        arrays["edge_drug_index"],
        patient_count,
        per_shard,
    )
    adj_entries, tile_index = _write_adjacency_shards(
        bundle_dir, partitions, source_checkpoint
    )
    _write_tile_edge_index(bundle_dir, tile_index)

    # Count total raw edges written
    total_raw_edges = sum(len(pairs) for pairs in partitions.values())

    # Build manifest
    all_shard_entries = (
        coord_entries + meta_entries + [drug_entry] + lod_entries
        + agg_edge_entries + adj_entries
    )

    vis_manifest = {
        "format": "tekarx.sharded_webgl_graph",
        "format_version": SHARD_FORMAT_VERSION,
        "exporter_version": EXPORTER_VERSION,
        "intended_use": "research exploration; not diagnosis",
        "layout": layout,
        "seed": seed,
        "target_shard_bytes": target_shard_bytes,
        "lod_levels": LOD_LEVELS,
        "counts": {
            "patients": patient_count,
            "drugs": drug_count,
            "edges": edge_count,
            "raw_edges_exported": total_raw_edges,
            "lod_cells": len(lod_cells),
            "aggregate_edges": len(aggregates),
            "serious_patients": int(np.count_nonzero(np.asarray(arrays["patient_y"]) == 1)),
        },
        "drug_metadata_path": drug_meta_path,
        "tile_edge_index_path": "edges/tile_index.json",
        "shards": [asdict(e) for e in all_shard_entries],
        "source": {
            "graph_manifest": str(manifest_path),
            "checkpoint": source_checkpoint,
            "graph_counts": {
                "patients": int(counts.get("patient_nodes", patient_count)),
                "drugs": drug_count,
                "edges": edge_count,
            },
        },
    }
    manifest_out = bundle_dir / "manifest.json"
    _write_json_atomic(manifest_out, vis_manifest)

    # Provenance
    provenance = {
        "graph_checkpoint_id": source_checkpoint,
        "graph_manifest_path": str(manifest_path),
        "layout_version": layout,
        "exporter_version": EXPORTER_VERSION,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    provenance_path = bundle_dir / "provenance.json"
    _write_json_atomic(provenance_path, provenance)

    # Phase E: Write HTML
    print("Full viz 6/6: writing WebGL2 full-graph launcher", flush=True)
    relative_manifest = Path(bundle_dir.name, "manifest.json").as_posix()
    _write_text_atomic(
        output_path,
        _render_full_webgl_html(manifest_url=relative_manifest),
    )

    # Record
    total_bytes = sum(e.byte_length for e in all_shard_entries)
    elapsed = time.monotonic() - wall_start

    record = FullVisualizationRecord(
        output_path=str(output_path),
        manifest_path=str(manifest_out),
        provenance_path=str(provenance_path),
        bundle_directory=str(bundle_dir),
        layout=layout,
        total_patients=patient_count,
        total_drugs=drug_count,
        total_edges=edge_count,
        patient_shard_count=len(coord_entries),
        edge_shard_count=len(adj_entries),
        lod_levels=LOD_LEVELS,
        total_bytes=total_bytes,
        seed=seed,
        elapsed_seconds=round(elapsed, 2),
        source_checkpoint=source_checkpoint,
    )

    # Write provenance record alongside HTML
    _write_json_atomic(
        output_path.with_suffix(".json"),
        {
            "dataset": "TekaRx full-graph sharded WebGL visualization",
            "intended_use": "research exploration; not diagnosis",
            "record": asdict(record),
        },
    )
    return record


# ---------------------------------------------------------------------------
# HTML template rendering (Phase E)
# ---------------------------------------------------------------------------


def _render_full_webgl_html(*, manifest_url: str) -> str:
    """Render the full-graph WebGL2 HTML template."""
    template = (
        files("tekarx").joinpath("templates", "graph_webgl_full.html").read_text(encoding="utf-8")
    )
    title = "TekaRx Full Patient-Drug Graph"
    return (
        template.replace("__TEKARX_TITLE__", title)
        .replace("__TEKARX_MANIFEST_URL__", json.dumps(manifest_url))
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _write_text_atomic(path: Path, value: str) -> None:
    """Write text atomically via temp-then-rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def _write_json_atomic(path: Path, value: Any, *, compact: bool = False) -> None:
    """Write JSON atomically."""
    serialized = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if compact
        else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _write_text_atomic(path, serialized)
