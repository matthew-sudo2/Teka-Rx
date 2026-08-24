from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tekarx.visualization import visualize_graph


def _array(root: Path, name: str, values: np.ndarray) -> dict[str, object]:
    path = root / f"{name}.npy"
    np.save(path, values, allow_pickle=False)
    return {"path": path.name, "dtype": str(values.dtype), "shape": list(values.shape)}


def test_visualize_graph_writes_bounded_standalone_html(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    arrays_dir = data_dir / "processed" / "tekarx_graph_arrays"
    arrays_dir.mkdir(parents=True)
    arrays = {
        "patient_primaryid": _array(
            arrays_dir, "patient_primaryid", np.array([100, 101, 102, 103], dtype=np.int64)
        ),
        "patient_y": _array(
            arrays_dir, "patient_y", np.array([0, 1, 0, 1], dtype=np.int8)
        ),
        "patient_split_id": _array(
            arrays_dir, "patient_split_id", np.array([0, 1, 1, 2], dtype=np.int8)
        ),
        "edge_patient_index": _array(
            arrays_dir,
            "edge_patient_index",
            np.array([0, 1, 1, 2, 2, 3], dtype=np.int32),
        ),
        "edge_drug_index": _array(
            arrays_dir,
            "edge_drug_index",
            np.array([0, 0, 1, 0, 2, 2], dtype=np.int32),
        ),
        "drug_x": _array(
            arrays_dir,
            "drug_x",
            np.array(
                [
                    [0.2, 1.0, 1.0, *([0.0] * 25)],
                    [0.1, 0.0, 0.0, 1.0, *([0.0] * 24)],
                    [0.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * 23)],
                ],
                dtype=np.float32,
            ),
        ),
    }
    pq.write_table(
        pa.table(
            {
                "node_index": [0, 1, 2],
                "semantic_id": [-1, -2, -3],
                "node_label": ["DRUG A", "DRUG B", "DRUG C"],
                "node_kind": ["unknown", "unknown", "unknown"],
            }
        ),
        arrays_dir / "drug_nodes.parquet",
    )
    manifest = {
        "format": "tekarx.memmap_graph",
        "format_version": 1,
        "arrays": arrays,
        "drug_metadata_path": "drug_nodes.parquet",
        "edge_order": {
            "split_offsets": {"train": [0, 1], "validation": [1, 5], "test": [5, 6]}
        },
        "counts": {"patient_nodes": 4, "drug_nodes": 3, "patient_drug_edges": 6},
    }
    (arrays_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    output = data_dir / "processed" / "sample.html"
    record = visualize_graph(
        data_dir=data_dir,
        split="validation",
        patients=2,
        top_drugs=2,
        seed=7,
        output=output,
    )

    rendered = output.read_text(encoding="utf-8")
    assert record.rendered_patients == 2
    assert record.rendered_drugs == 2
    assert record.rendered_edges == 3
    assert record.serious_patients == 1
    assert "DRUG A" in rendered
    assert '"primaryid": "101"' in rendered
    assert "<script>" in rendered
    assert output.with_suffix(".json").is_file()

    output_3d = data_dir / "processed" / "sample_3d.html"
    record_3d = visualize_graph(
        data_dir=data_dir,
        split="validation",
        layout="3d",
        patients=2,
        top_drugs=2,
        seed=7,
        output=output_3d,
    )
    rendered_3d = output_3d.read_text(encoding="utf-8")
    assert record_3d.layout == "3d"
    assert "<canvas" in rendered_3d
    assert "Drag to rotate" in rendered_3d
    assert "Top drugs" in rendered_3d
    assert "topDrugLabels" in rendered_3d
