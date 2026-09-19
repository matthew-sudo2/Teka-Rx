from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tekarx.cli import main as cli_main
from tekarx.full_visualization import (
    FullVisualizationError,
    estimate_full_visualization,
    export_full_visualization,
)


def _array(root: Path, name: str, values: np.ndarray) -> dict[str, object]:
    path = root / f"{name}.npy"
    np.save(path, values, allow_pickle=False)
    return {"path": path.name, "dtype": str(values.dtype), "shape": list(values.shape)}


def _setup_synthetic_graph(tmp_path: Path, patient_count: int = 20, drug_count: int = 5) -> Path:
    data_dir = tmp_path / "data"
    arrays_dir = data_dir / "processed" / "tekarx_graph_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = np.arange(100, 100 + patient_count, dtype=np.int64)
    patient_y = np.array([i % 2 for i in range(patient_count)], dtype=np.int8)
    patient_split = np.array([i % 3 for i in range(patient_count)], dtype=np.int8)

    # Create edges: each patient linked to 2 drugs
    edge_p: list[int] = []
    edge_d: list[int] = []
    for p in range(patient_count):
        edge_p.extend([p, p])
        edge_d.extend([p % drug_count, (p + 1) % drug_count])

    edge_p_arr = np.array(edge_p, dtype=np.int32)
    edge_d_arr = np.array(edge_d, dtype=np.int32)

    drug_x_vals = []
    for d in range(drug_count):
        row = [0.1 * (d + 1), 1.0 if d % 2 == 0 else 0.0, 1.0 if d == 0 else 0.0, *([0.0] * 25)]
        drug_x_vals.append(row)
    drug_x = np.array(drug_x_vals, dtype=np.float32)

    arrays = {
        "patient_primaryid": _array(arrays_dir, "patient_primaryid", patient_ids),
        "patient_y": _array(arrays_dir, "patient_y", patient_y),
        "patient_split_id": _array(arrays_dir, "patient_split_id", patient_split),
        "edge_patient_index": _array(arrays_dir, "edge_patient_index", edge_p_arr),
        "edge_drug_index": _array(arrays_dir, "edge_drug_index", edge_d_arr),
        "drug_x": _array(arrays_dir, "drug_x", drug_x),
    }

    pq.write_table(
        pa.table(
            {
                "node_index": list(range(drug_count)),
                "semantic_id": [-1 - i for i in range(drug_count)],
                "node_label": [f"DRUG_{chr(65+i)}" for i in range(drug_count)],
                "node_kind": ["mapped" for _ in range(drug_count)],
            }
        ),
        arrays_dir / "drug_nodes.parquet",
    )

    manifest = {
        "format": "tekarx.memmap_graph",
        "format_version": 1,
        "arrays": arrays,
        "drug_metadata_path": "drug_nodes.parquet",
        "counts": {
            "patient_nodes": patient_count,
            "drug_nodes": drug_count,
            "patient_drug_edges": len(edge_p),
        },
    }
    (arrays_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data_dir


def test_estimate_full_visualization(tmp_path: Path) -> None:
    data_dir = _setup_synthetic_graph(tmp_path, patient_count=50, drug_count=10)
    estimate = estimate_full_visualization(data_dir=data_dir, target_shard_bytes=1024)

    assert estimate.graph_patients == 50
    assert estimate.graph_drugs == 10
    assert estimate.graph_edges == 100
    assert estimate.estimated_total_bytes > 0
    assert estimate.estimated_shard_count > 1
    assert estimate.estimated_cpu_working_set_bytes > 0


def test_export_full_visualization_creates_sharded_bundle(tmp_path: Path) -> None:
    data_dir = _setup_synthetic_graph(tmp_path, patient_count=20, drug_count=5)
    output_html = data_dir / "processed" / "full_viz.html"

    # Use small target shard size to trigger multiple patient shards
    record = export_full_visualization(
        data_dir=data_dir,
        target_shard_bytes=128,  # Small shard size (10 patients max per coord shard)
        seed=42,
        output=output_html,
    )

    assert record.total_patients == 20
    assert record.total_drugs == 5
    assert record.total_edges == 40
    assert record.patient_shard_count >= 2
    assert record.edge_shard_count >= 1
    assert record.total_bytes > 0
    assert record.elapsed_seconds >= 0.0

    # HTML launcher checks
    assert output_html.is_file()
    html_text = output_html.read_text(encoding="utf-8")
    assert "TekaRx Full Patient-Drug Graph" in html_text
    assert "full_viz/manifest.json" in html_text
    assert "<canvas" in html_text

    # Manifest checks
    bundle_dir = output_html.parent / "full_viz"
    manifest_path = bundle_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_data["format"] == "tekarx.sharded_webgl_graph"
    assert manifest_data["format_version"] == 1
    assert manifest_data["counts"]["patients"] == 20
    assert manifest_data["counts"]["drugs"] == 5
    assert manifest_data["counts"]["edges"] == 40
    assert manifest_data["counts"]["lod_cells"] > 0
    assert len(manifest_data["shards"]) > 0

    # Shard entries check
    logical_types = {s["logical_type"] for s in manifest_data["shards"]}
    assert "patient_coordinates" in logical_types
    assert "patient_metadata" in logical_types
    assert "drug_coordinates" in logical_types
    assert "lod_l0_nodes" in logical_types
    assert "raw_adjacency" in logical_types

    # Provenance file check
    provenance_path = bundle_dir / "provenance.json"
    assert provenance_path.is_file()
    prov_data = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert prov_data["seed"] == 42
    assert "timestamp" in prov_data

    # Check sidecar record JSON
    sidecar_json = output_html.with_suffix(".json")
    assert sidecar_json.is_file()


def test_full_visualization_resume(tmp_path: Path) -> None:
    data_dir = _setup_synthetic_graph(tmp_path, patient_count=20, drug_count=5)
    output_html = data_dir / "processed" / "resume_test.html"

    # First export
    export_full_visualization(
        data_dir=data_dir,
        target_shard_bytes=128,
        seed=42,
        output=output_html,
    )

    # Second export with resume=True
    record_resume = export_full_visualization(
        data_dir=data_dir,
        target_shard_bytes=128,
        seed=42,
        output=output_html,
        resume=True,
    )
    assert record_resume.total_patients == 20


def test_full_visualization_determinism(tmp_path: Path) -> None:
    data_dir = _setup_synthetic_graph(tmp_path, patient_count=30, drug_count=6)
    out1 = data_dir / "processed" / "run1.html"
    out2 = data_dir / "processed" / "run2.html"

    rec1 = export_full_visualization(data_dir=data_dir, seed=123, output=out1)
    rec2 = export_full_visualization(data_dir=data_dir, seed=123, output=out2)

    m1 = json.loads((Path(rec1.bundle_directory) / "manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((Path(rec2.bundle_directory) / "manifest.json").read_text(encoding="utf-8"))

    s1 = [s["sha256"] for s in m1["shards"]]
    s2 = [s["sha256"] for s in m2["shards"]]
    assert s1 == s2


def test_full_visualization_provenance_validation_rejects_mismatch(tmp_path: Path) -> None:
    data_dir = _setup_synthetic_graph(tmp_path, patient_count=10, drug_count=2)
    arrays_dir = data_dir / "processed" / "tekarx_graph_arrays"

    # Corrupt patient_y array shape in manifest
    manifest_path = arrays_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["arrays"]["patient_y"]["shape"] = [999]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FullVisualizationError, match="does not match its manifest"):
        export_full_visualization(data_dir=data_dir)


def test_cli_estimate_and_export_full_visualization(tmp_path: Path) -> None:
    data_dir = _setup_synthetic_graph(tmp_path, patient_count=15, drug_count=3)
    out_html = data_dir / "processed" / "cli_full.html"

    # Test estimate-visualization CLI
    ret_est = cli_main(
        [
            "estimate-visualization",
            "--data-dir",
            str(data_dir),
            "--target-shard-size",
            "16MB",
        ]
    )
    assert ret_est == 0

    # Test visualize-graph --full CLI
    ret_exp = cli_main(
        [
            "visualize-graph",
            "--data-dir",
            str(data_dir),
            "--full",
            "--target-shard-size",
            "16MB",
            "--output",
            str(out_html),
        ]
    )
    assert ret_exp == 0
    assert out_html.is_file()
