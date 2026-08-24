from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "visualize_full_graph_colab.ipynb"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_full_graph_colab_notebook_is_clean_and_compilable() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        compile(_source(cell), f"{NOTEBOOK_PATH.name}:{cell.get('id')}", "exec")


def test_full_graph_colab_restores_only_visualization_projection() -> None:
    notebook = _notebook()
    code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    required = (
        "drive.mount",
        "/content/drive/MyDrive/Teka-Rx-full/data",
        "_GRAPH_SUCCESS.json",
        "verified_complete",
        "gnn-full",
        "graph_checkpoints",
        "graph_visualization_checkpoints",
        "patient_primaryid",
        "patient_y",
        "patient_split_id",
        "edge_patient_index",
        "edge_drug_index",
        "drug_x",
        "copy_with_progress",
        "visualize-graph",
        '"--layout", "3d"',
        "graph_patient_nodes",
        "graph_edges",
        "serve_kernel_port_as_iframe",
        "sha256_file",
    )
    for marker in required:
        assert marker in code

    restore_cell = next(
        cell for cell in notebook["cells"] if cell.get("id") == "restore-topology"
    )
    restore_source = _source(restore_cell)
    assert '"patient_x"' not in restore_source.split("intentionally_omitted")[0]
    assert '"patient_x"' in restore_source.split("intentionally_omitted")[1]
    assert "rmtree" not in code
    assert "--evaluate-test" not in code
    assert "github_pat_" not in code
