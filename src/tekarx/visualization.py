"""Memory-bounded binary export for GPU patient-drug graph visualization."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
MAX_VISUALIZATION_PATIENTS = 1_000_000
MAX_VISUALIZATION_DRUGS = 100_000
EDGE_SCAN_CHUNK_SIZE = 1_000_000
PATIENT_WRITE_CHUNK_SIZE = 1_000_000
REQUIRED_ARRAYS = (
    "patient_primaryid",
    "patient_y",
    "patient_split_id",
    "edge_patient_index",
    "edge_drug_index",
)


class GraphVisualizationError(RuntimeError):
    """Raised when a graph cannot be safely sampled or rendered."""


@dataclass(frozen=True)
class GraphVisualizationRecord:
    """Summary of an exported browser visualization."""

    output_path: str
    array_manifest_path: str
    visualization_manifest_path: str
    asset_directory: str
    split: str
    layout: str
    requested_patients: int
    rendered_patients: int
    rendered_drugs: int
    rendered_edges: int
    serious_patients: int
    nonserious_patients: int
    graph_patient_nodes: int
    graph_drug_nodes: int
    graph_edges: int
    binary_bytes: int
    seed: int


def visualize_graph(
    *,
    data_dir: Path,
    graph_dir: Path | None = None,
    split: str = "validation",
    layout: str = "3d",
    patients: int = 100,
    top_drugs: int = 50,
    seed: int = 42,
    output: Path | None = None,
) -> GraphVisualizationRecord:
    """Export a static-layout graph as typed binary buffers plus WebGL2 HTML.

    Source graph arrays remain memory-mapped. Patient selection and edge
    remapping are vectorized in bounded chunks, and the browser receives no
    million-element JSON structures.
    """
    if split not in SPLIT_IDS:
        raise ValueError(f"split must be one of {tuple(SPLIT_IDS)}")
    if layout not in ("2d", "3d"):
        raise ValueError("layout must be '2d' or '3d'")
    if not 1 <= patients <= MAX_VISUALIZATION_PATIENTS:
        raise ValueError(f"patients must be between 1 and {MAX_VISUALIZATION_PATIENTS}")
    if not 1 <= top_drugs <= MAX_VISUALIZATION_DRUGS:
        raise ValueError(f"top_drugs must be between 1 and {MAX_VISUALIZATION_DRUGS}")

    data_root = Path(data_dir).resolve()
    source_manifest_path = _resolve_array_manifest(data_root, graph_dir)
    source_manifest = _read_manifest(source_manifest_path)
    arrays = {
        name: _load_array(source_manifest_path, source_manifest, name) for name in REQUIRED_ARRAYS
    }
    drug_x = _load_array(source_manifest_path, source_manifest, "drug_x", required=False)
    patient_count = _validate_source_arrays(arrays)
    counts = source_manifest.get("counts", {})
    graph_drug_count = int(counts.get("drug_nodes", int(np.max(arrays["edge_drug_index"])) + 1))
    if graph_drug_count < 1:
        raise GraphVisualizationError("source graph has no drug nodes")

    print("Visualization 1/5: selecting patients with bounded memory", flush=True)
    selected = _sample_patient_indices(
        arrays["patient_split_id"],
        arrays["patient_y"],
        split_id=SPLIT_IDS[split],
        count=patients,
        seed=seed,
    )
    print("Visualization 2/5: scanning selected exposure frequencies", flush=True)
    frequencies = _selected_drug_frequencies(
        arrays["edge_patient_index"],
        arrays["edge_drug_index"],
        selected,
        graph_drug_count=graph_drug_count,
        manifest=source_manifest,
        split=split,
    )
    kept_drugs = _highest_frequency_drugs(frequencies, top_drugs)
    if kept_drugs.size == 0:
        raise GraphVisualizationError(f"sampled {split} patients have no drug edges")

    default_name = "graph_visualization_3d.html" if layout == "3d" else "graph_visualization.html"
    output_path = Path(output or data_root / "processed" / default_name)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asset_directory = output_path.parent / f"{output_path.stem}_assets"
    asset_directory.mkdir(parents=True, exist_ok=True)

    print("Visualization 3/5: writing offline coordinates and metadata", flush=True)
    patient_coords_path = asset_directory / "patient_coords.bin"
    drug_coords_path = asset_directory / "drug_coords.bin"
    patient_y_path = asset_directory / "patient_y.bin"
    patient_split_path = asset_directory / "patient_split_id.bin"
    _write_patient_coordinates(
        patient_coords_path,
        arrays["patient_y"],
        selected,
        seed=seed,
        layout=layout,
    )
    _write_drug_coordinates(drug_coords_path, kept_drugs.size, layout=layout)
    _write_selected_uint8(patient_y_path, arrays["patient_y"], selected)
    _write_selected_uint8(patient_split_path, arrays["patient_split_id"], selected)

    drug_global_to_local = np.full(graph_drug_count, -1, dtype=np.int32)
    drug_global_to_local[kept_drugs] = np.arange(kept_drugs.size, dtype=np.int32)
    rendered_edge_count = int(frequencies[kept_drugs].sum(dtype=np.int64))
    print(
        f"Visualization 4/5: streaming {rendered_edge_count:,} remapped edges",
        flush=True,
    )
    edges_path = asset_directory / "edges.bin"
    _write_edge_indices(
        edges_path,
        arrays["edge_patient_index"],
        arrays["edge_drug_index"],
        selected,
        drug_global_to_local,
        expected_edges=rendered_edge_count,
        manifest=source_manifest,
        split=split,
    )

    drug_metadata = _read_drug_metadata(source_manifest_path, source_manifest)
    drug_names = _drugcentral_names(data_root, set(map(int, kept_drugs)), drug_metadata)
    rendered_drug_metadata = _rendered_drug_metadata(
        kept_drugs, frequencies, drug_metadata, drug_names, drug_x
    )
    drug_metadata_path = asset_directory / "drug_metadata.json"
    _write_json_atomic(drug_metadata_path, rendered_drug_metadata, compact=True)

    assets = {
        "patient_coords": _binary_metadata(patient_coords_path, "float32", [selected.size, 3]),
        "drug_coords": _binary_metadata(drug_coords_path, "float32", [kept_drugs.size, 3]),
        "patient_y": _binary_metadata(patient_y_path, "uint8", [selected.size]),
        "patient_split_id": _binary_metadata(patient_split_path, "uint8", [selected.size]),
        "edges": _binary_metadata(edges_path, "int32", [rendered_edge_count, 2]),
    }
    visualization_manifest_path = asset_directory / "manifest.json"
    visualization_manifest = {
        "format": "tekarx.webgl_graph",
        "format_version": 1,
        "intended_use": "research exploration; not diagnosis",
        "split": split,
        "layout": layout,
        "seed": seed,
        "counts": {
            "patients": int(selected.size),
            "drugs": int(kept_drugs.size),
            "edges": rendered_edge_count,
            "serious_patients": int(
                np.count_nonzero(np.asarray(arrays["patient_y"])[selected] == 1)
            ),
        },
        "arrays": assets,
        "edge_encoding": ("row-major [patient_local_index, patient_count + drug_local_index]"),
        "drug_metadata_path": drug_metadata_path.name,
        "source": {
            "graph_manifest": str(source_manifest_path),
            "graph_counts": {
                "patients": int(counts.get("patient_nodes", patient_count)),
                "drugs": graph_drug_count,
                "edges": int(
                    counts.get("patient_drug_edges", arrays["edge_patient_index"].shape[0])
                ),
            },
        },
    }
    _write_json_atomic(visualization_manifest_path, visualization_manifest, compact=True)

    relative_manifest = Path(
        Path(asset_directory).relative_to(output_path.parent), "manifest.json"
    ).as_posix()
    print("Visualization 5/5: writing WebGL2 launcher and provenance", flush=True)
    _write_text_atomic(
        output_path,
        _render_webgl_html(split=split, layout=layout, manifest_url=relative_manifest),
    )

    serious_count = int(visualization_manifest["counts"]["serious_patients"])
    binary_bytes = sum(
        path.stat().st_size
        for path in (
            patient_coords_path,
            drug_coords_path,
            patient_y_path,
            patient_split_path,
            edges_path,
        )
    )
    record = GraphVisualizationRecord(
        output_path=str(output_path),
        array_manifest_path=str(source_manifest_path),
        visualization_manifest_path=str(visualization_manifest_path),
        asset_directory=str(asset_directory),
        split=split,
        layout=layout,
        requested_patients=patients,
        rendered_patients=int(selected.size),
        rendered_drugs=int(kept_drugs.size),
        rendered_edges=rendered_edge_count,
        serious_patients=serious_count,
        nonserious_patients=int(selected.size) - serious_count,
        graph_patient_nodes=int(counts.get("patient_nodes", patient_count)),
        graph_drug_nodes=graph_drug_count,
        graph_edges=int(counts.get("patient_drug_edges", arrays["edge_patient_index"].shape[0])),
        binary_bytes=binary_bytes,
        seed=seed,
    )
    _write_json_atomic(
        output_path.with_suffix(".json"),
        {
            "dataset": "TekaRx patient-drug WebGL visualization",
            "intended_use": "research exploration; not diagnosis",
            "sampling": "seeded systematic, approximately class-balanced patient sample",
            "record": asdict(record),
        },
    )
    return record


def _validate_source_arrays(arrays: dict[str, np.ndarray]) -> int:
    patient_count = int(arrays["patient_y"].shape[0])
    if arrays["patient_primaryid"].shape != (patient_count,):
        raise GraphVisualizationError("patient ID and target arrays have different lengths")
    if arrays["patient_split_id"].shape != (patient_count,):
        raise GraphVisualizationError("patient split and target arrays have different lengths")
    if arrays["edge_patient_index"].shape != arrays["edge_drug_index"].shape:
        raise GraphVisualizationError("patient and drug edge arrays have different lengths")
    return patient_count


def _sample_patient_indices(
    split_ids: np.ndarray,
    labels: np.ndarray,
    *,
    split_id: int,
    count: int,
    seed: int,
) -> np.ndarray:
    """Return a deterministic, systematic, approximately balanced sample.

    This uses two chunked passes and O(sample-size) memory. It avoids a Python
    set with one million integer objects and does not materialize all eligible
    patient indices.
    """
    capacities = np.zeros(2, dtype=np.int64)
    for start in range(0, split_ids.shape[0], PATIENT_WRITE_CHUNK_SIZE):
        stop = min(start + PATIENT_WRITE_CHUNK_SIZE, split_ids.shape[0])
        chunk_splits = np.asarray(split_ids[start:stop])
        chunk_labels = np.asarray(labels[start:stop])
        eligible = chunk_splits == split_id
        capacities += np.bincount(chunk_labels[eligible].astype(np.int64, copy=False), minlength=2)[
            :2
        ]
    available = int(capacities.sum())
    if available == 0:
        raise GraphVisualizationError("selected graph split has no patients")
    requested = min(count, available)
    targets = np.array([requested - requested // 2, requested // 2], dtype=np.int64)
    targets = np.minimum(targets, capacities)
    remaining = requested - int(targets.sum())
    for label in np.argsort(-(capacities - targets)):
        addition = min(remaining, int(capacities[label] - targets[label]))
        targets[label] += addition
        remaining -= addition
        if remaining == 0:
            break

    rng = np.random.default_rng(seed)
    desired_ranks: list[np.ndarray] = []
    for label in (0, 1):
        target = int(targets[label])
        capacity = int(capacities[label])
        if target == 0:
            desired_ranks.append(np.empty(0, dtype=np.int64))
            continue
        phase = float(rng.random())
        ranks = np.floor((np.arange(target, dtype=np.float64) + phase) * capacity / target).astype(
            np.int64
        )
        desired_ranks.append(np.minimum(ranks, capacity - 1))

    selected_by_label = [np.empty(int(targets[label]), dtype=np.int64) for label in (0, 1)]
    written = np.zeros(2, dtype=np.int64)
    seen = np.zeros(2, dtype=np.int64)
    for start in range(0, split_ids.shape[0], PATIENT_WRITE_CHUNK_SIZE):
        stop = min(start + PATIENT_WRITE_CHUNK_SIZE, split_ids.shape[0])
        chunk_splits = np.asarray(split_ids[start:stop])
        chunk_labels = np.asarray(labels[start:stop])
        for label in (0, 1):
            local = np.flatnonzero((chunk_splits == split_id) & (chunk_labels == label))
            first_rank = int(seen[label])
            last_rank = first_rank + int(local.size)
            ranks = desired_ranks[label]
            left = int(np.searchsorted(ranks, first_rank, side="left"))
            right = int(np.searchsorted(ranks, last_rank, side="left"))
            take = ranks[left:right] - first_rank
            destination = int(written[label])
            selected_by_label[label][destination : destination + take.size] = local[take] + start
            written[label] += take.size
            seen[label] = last_rank
    if any(int(written[label]) != int(targets[label]) for label in (0, 1)):
        raise GraphVisualizationError("patient sampler did not fill its allocated output")
    return np.sort(np.concatenate(selected_by_label))


def _selected_edge_chunks(
    edge_patients: np.ndarray,
    edge_drugs: np.ndarray,
    selected: np.ndarray,
    *,
    manifest: dict[str, Any],
    split: str,
    description: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    offsets = manifest.get("edge_order", {}).get("split_offsets", {}).get(split)
    start, stop = (
        (int(offsets[0]), int(offsets[1])) if offsets else (0, int(edge_patients.shape[0]))
    )
    with tqdm(
        total=stop - start,
        desc=description,
        unit="edges",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for begin in range(start, stop, EDGE_SCAN_CHUNK_SIZE):
            end = min(begin + EDGE_SCAN_CHUNK_SIZE, stop)
            patients = np.asarray(edge_patients[begin:end], dtype=np.int64)
            positions = np.searchsorted(selected, patients)
            in_bounds = positions < selected.size
            matched = np.zeros(patients.size, dtype=bool)
            matched[in_bounds] = selected[positions[in_bounds]] == patients[in_bounds]
            if np.any(matched):
                yield (
                    positions[matched].astype(np.int32, copy=False),
                    np.asarray(edge_drugs[begin:end])[matched].astype(np.int64, copy=False),
                )
            progress.update(end - begin)


def _selected_drug_frequencies(
    edge_patients: np.ndarray,
    edge_drugs: np.ndarray,
    selected: np.ndarray,
    *,
    graph_drug_count: int,
    manifest: dict[str, Any],
    split: str,
) -> np.ndarray:
    frequencies = np.zeros(graph_drug_count, dtype=np.int64)
    for _patient_local, drugs in _selected_edge_chunks(
        edge_patients,
        edge_drugs,
        selected,
        manifest=manifest,
        split=split,
        description="Exposure pass 1/2",
    ):
        if np.any(drugs < 0) or np.any(drugs >= graph_drug_count):
            raise GraphVisualizationError("edge references a drug outside graph bounds")
        frequencies += np.bincount(drugs, minlength=graph_drug_count)
    return frequencies


def _highest_frequency_drugs(frequencies: np.ndarray, limit: int) -> np.ndarray:
    present = np.flatnonzero(frequencies)
    if present.size == 0:
        return present.astype(np.int64)
    order = np.lexsort((present, -frequencies[present]))
    return present[order[:limit]].astype(np.int64, copy=False)


def _write_edge_indices(
    path: Path,
    edge_patients: np.ndarray,
    edge_drugs: np.ndarray,
    selected: np.ndarray,
    drug_global_to_local: np.ndarray,
    *,
    expected_edges: int,
    manifest: dict[str, Any],
    split: str,
) -> None:
    temporary, output = _open_binary_memmap(path, (expected_edges, 2), np.dtype("<i4"))
    cursor = 0
    try:
        for patient_local, drugs in _selected_edge_chunks(
            edge_patients,
            edge_drugs,
            selected,
            manifest=manifest,
            split=split,
            description="Exposure pass 2/2",
        ):
            drug_local = drug_global_to_local[drugs]
            keep = drug_local >= 0
            amount = int(np.count_nonzero(keep))
            if amount:
                output[cursor : cursor + amount, 0] = patient_local[keep]
                output[cursor : cursor + amount, 1] = selected.size + drug_local[keep]
                cursor += amount
        if cursor != expected_edges:
            raise GraphVisualizationError(
                f"edge export count changed between passes: {cursor} != {expected_edges}"
            )
        output.flush()
    finally:
        del output
    temporary.replace(path)


def _write_patient_coordinates(
    path: Path,
    labels: np.ndarray,
    selected: np.ndarray,
    *,
    seed: int,
    layout: str,
) -> None:
    temporary, output = _open_binary_memmap(path, (selected.size, 3), np.dtype("<f4"))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    try:
        for start in range(0, selected.size, PATIENT_WRITE_CHUNK_SIZE):
            stop = min(start + PATIENT_WRITE_CHUNK_SIZE, selected.size)
            ordinal = np.arange(start, stop, dtype=np.float64)
            fraction = (ordinal + 0.5) / max(1, selected.size)
            radius = 1.48 * np.sqrt(fraction)
            angle = (ordinal + (seed % 104729)) * golden_angle
            patient_labels = np.asarray(labels[selected[start:stop]], dtype=np.float32)
            output[start:stop, 0] = -1.18 + (patient_labels - 0.5) * 0.12
            output[start:stop, 1] = radius * np.cos(angle)
            output[start:stop, 2] = radius * np.sin(angle) if layout == "3d" else 0.0
        output.flush()
    finally:
        del output
    temporary.replace(path)


def _write_drug_coordinates(path: Path, count: int, *, layout: str) -> None:
    temporary, output = _open_binary_memmap(path, (count, 3), np.dtype("<f4"))
    ordinal = np.arange(count, dtype=np.float64)
    fraction = (ordinal + 0.5) / max(1, count)
    angle = ordinal * np.pi * (3.0 - np.sqrt(5.0))
    radius = 1.10 * np.sqrt(fraction)
    try:
        output[:, 0] = 1.18
        output[:, 1] = radius * np.cos(angle)
        output[:, 2] = radius * np.sin(angle) if layout == "3d" else 0.0
        output.flush()
    finally:
        del output
    temporary.replace(path)


def _write_selected_uint8(path: Path, source: np.ndarray, selected: np.ndarray) -> None:
    temporary, output = _open_binary_memmap(path, (selected.size,), np.dtype("u1"))
    try:
        for start in range(0, selected.size, PATIENT_WRITE_CHUNK_SIZE):
            stop = min(start + PATIENT_WRITE_CHUNK_SIZE, selected.size)
            output[start:stop] = np.asarray(source[selected[start:stop]], dtype=np.uint8)
        output.flush()
    finally:
        del output
    temporary.replace(path)


def _open_binary_memmap(
    path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]
) -> tuple[Path, np.memmap]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.unlink(missing_ok=True)
        output = np.memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    except (OSError, ValueError) as exc:
        raise GraphVisualizationError(
            f"cannot allocate binary visualization asset: {path}"
        ) from exc
    return temporary, output


def _binary_metadata(path: Path, dtype: str, shape: list[int]) -> dict[str, str | int | list[int]]:
    return {"path": path.name, "dtype": dtype, "shape": shape, "bytes": path.stat().st_size}


def _rendered_drug_metadata(
    kept_drugs: np.ndarray,
    frequencies: np.ndarray,
    metadata: dict[int, dict[str, Any]],
    names: dict[int, str],
    drug_x: np.ndarray | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for local_index, raw_global_index in enumerate(kept_drugs):
        global_index = int(raw_global_index)
        source = metadata.get(global_index, {})
        features = _describe_drug_features(drug_x, global_index)
        result.append(
            {
                "index": local_index,
                "source_index": global_index,
                "label": names.get(global_index)
                or str(source.get("node_label", f"Drug {global_index}")),
                "semantic_id": int(source.get("semantic_id", 0)),
                "kind": str(source.get("node_kind", "unknown")),
                "degree": int(frequencies[global_index]),
                **features,
            }
        )
    return result


def _resolve_array_manifest(data_dir: Path, graph_dir: Path | None) -> Path:
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
        if candidate.is_file() and _has_required_arrays(candidate):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise GraphVisualizationError(f"no complete visualization graph found; searched: {searched}")


def _has_required_arrays(manifest_path: Path) -> bool:
    try:
        manifest = _read_manifest(manifest_path)
        for name in REQUIRED_ARRAYS:
            metadata = manifest.get("arrays", {}).get(name, {})
            relative = metadata.get("path")
            if not isinstance(relative, str) or not (manifest_path.parent / relative).is_file():
                return False
        metadata_path = manifest.get("drug_metadata_path")
        return isinstance(metadata_path, str) and (manifest_path.parent / metadata_path).is_file()
    except GraphVisualizationError:
        return False


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphVisualizationError(f"cannot read graph array manifest: {path}") from exc
    if manifest.get("format") != "tekarx.memmap_graph":
        raise GraphVisualizationError(
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
    metadata = manifest.get("arrays", {}).get(name)
    if not isinstance(metadata, dict):
        if required:
            raise GraphVisualizationError(f"graph manifest has no {name!r} array")
        return None
    relative = metadata.get("path")
    if not isinstance(relative, str):
        raise GraphVisualizationError(f"graph array {name!r} has no path")
    root = manifest_path.parent.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        if not required:
            return None
        raise GraphVisualizationError(f"missing graph array {name!r}: {path}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise GraphVisualizationError(f"cannot memory-map graph array {name!r}: {path}") from exc
    if str(array.dtype) != metadata.get("dtype") or list(array.shape) != metadata.get("shape"):
        raise GraphVisualizationError(f"graph array {name!r} does not match its manifest")
    return array


def _read_drug_metadata(manifest_path: Path, manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    relative = manifest.get("drug_metadata_path")
    if not isinstance(relative, str):
        raise GraphVisualizationError("graph manifest has no drug_metadata_path")
    path = (manifest_path.parent / relative).resolve()
    if not path.is_file():
        raise GraphVisualizationError(f"missing drug metadata: {path}")
    table = pq.read_table(path, columns=["node_index", "semantic_id", "node_label", "node_kind"])
    return {int(row["node_index"]): row for row in table.to_pylist()}


def _drugcentral_names(
    data_dir: Path,
    indices: set[int],
    metadata: dict[int, dict[str, Any]],
) -> dict[int, str]:
    structures = data_dir / "interim" / "drugcentral" / "structures.parquet"
    if not structures.is_file():
        return {}
    wanted = {
        str(metadata[index].get("semantic_id")): index
        for index in indices
        if metadata.get(index, {}).get("node_kind") == "mapped"
    }
    if not wanted:
        return {}
    table = pq.read_table(structures, columns=["id", "name"])
    result: dict[int, str] = {}
    for row in table.to_pylist():
        index = wanted.get(str(row["id"]))
        if index is not None and row.get("name"):
            result[index] = str(row["name"])
    return result


def _describe_drug_features(drug_x: np.ndarray | None, index: int) -> dict[str, Any]:
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


def _render_webgl_html(*, split: str, layout: str, manifest_url: str) -> str:
    template = files("tekarx").joinpath("templates", "graph_webgl.html").read_text(encoding="utf-8")
    title = f"TekaRx {split.title()} Patient-Drug Graph"
    if layout == "3d":
        title += " - 3D"
    return (
        template.replace("__TEKARX_TITLE__", title)
        .replace("__TEKARX_MANIFEST_URL__", json.dumps(manifest_url))
        .replace("__TEKARX_LAYOUT__", json.dumps(layout))
    )


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any, *, compact: bool = False) -> None:
    serialized = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if compact
        else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _write_text_atomic(path, serialized)
