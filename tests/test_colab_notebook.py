from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "train_full_colab.ipynb"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source(notebook: dict[str, object], cell_id: str) -> str:
    for cell in notebook["cells"]:
        if cell.get("id") == cell_id:
            return _source(cell)
    raise AssertionError(f"missing notebook cell: {cell_id}")


def test_colab_notebook_is_clean_and_portable() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"

    cells = notebook["cells"]
    assert any(cell["cell_type"] == "markdown" for cell in cells)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        compile(_source(cell), f"{NOTEBOOK_PATH.name}:{cell.get('id', 'cell')}", "exec")


def test_colab_notebook_preserves_storage_and_leakage_protocol() -> None:
    notebook = _notebook()
    code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    required = (
        "drive.mount",
        "/content/drive/MyDrive/Teka-Rx-full/data",
        "/content/tekarx-data",
        "gnn-full",
        "EXPECTED_QUARTERS",
        "len(EXPECTED_QUARTERS) != 22",
        "build-cohort",
        "build-drug-dictionary",
        "add-tabular-features",
        "feature-rescue",
        "faers_gnn-full.json",
        "--split-preset",
        "--skip-graph",
        "build-graph",
        "--graph-storage",
        "memory-mapped",
        "--xgb-device",
        "tekarx_graph_arrays",
        "graph_checkpoints",
        "gnn_checkpoints",
        "gnn_training_checkpoints",
        "_GRAPH_SUCCESS.json",
        "train-gnn",
        "--device",
        "cuda",
        "--feature-track",
        "prospective",
        "quarter_coverage",
        "held_out_patient_messages_to_shared_drugs",
        "included_in_patient_x",
        "test_evaluated",
        "test_auc",
    )
    for marker in required:
        assert marker in code

    install_project = _cell_source(notebook, "install-project")
    assert 'repo_source = str((REPO_DIR / "src").resolve())' in install_project
    assert "sys.path.insert(0, repo_source)" in install_project
    assert "import tekarx" in install_project

    forbidden = (
        "--evaluate-test",
        "completed_report",
        "gnn-small",
        "--graph-storage legacy",
        "rm -rf",
        "rmtree(DRIVE_DATA",
        "github_pat_",
        "AIza",
    )
    for marker in forbidden:
        assert marker not in code

    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert _cell_source(notebook, "train-gnn").count('"train-gnn"') == 1
    assert cell_ids.index("build-features") < cell_ids.index("build-graph")
    assert cell_ids.index("build-graph") < cell_ids.index("train-gnn")
    assert cell_ids.index("verify-graph") < cell_ids.index("train-gnn")


def test_colab_notebook_uses_restart_safe_versioned_graph_and_model_checkpoints() -> None:
    notebook = _notebook()
    verify_features = _cell_source(notebook, "verify-features")
    build_graph = _cell_source(notebook, "build-graph")
    verify_graph = _cell_source(notebook, "verify-graph")
    restore_graph = _cell_source(notebook, "restore-graph")
    train_gnn = _cell_source(notebook, "train-gnn")
    persist_model = _cell_source(notebook, "persist-model")

    # Existing durable markers remain valid until a fully verified replacement is published.
    assert "_GRAPH_SUCCESS.json" not in verify_features
    assert "_GNN_SUCCESS.json" not in verify_features
    assert "_GRAPH_SUCCESS.json" not in build_graph
    assert "_GNN_SUCCESS.json" not in build_graph
    assert ".unlink(" not in train_gnn

    assert 'drive_processed / "graph_checkpoints"' in verify_graph
    assert 'f".uploading-{graph_checkpoint_id}"' in verify_graph
    assert '"checkpoint_dir"' in verify_graph
    assert '"artifacts"' in verify_graph
    assert '"sha256"' in verify_graph
    assert verify_graph.index("os.replace(staging_checkpoint, final_checkpoint)") < (
        verify_graph.index('write_json_atomic(drive_processed / "_GRAPH_SUCCESS.json"')
    )
    assert verify_graph.index('write_json_atomic(drive_processed / "_GRAPH_SUCCESS.json"') < (
        verify_graph.index("for candidate in graph_checkpoint_root.iterdir()")
    )

    # A fresh GPU kernel can rerun setup and this restore cell without Cell 20 state.
    assert "from tekarx.transform.graph_storage import load_graph_arrays" in restore_graph
    assert "graph_copy_pairs = [" in restore_graph
    assert "require_local_capacity(graph_copy_pairs, headroom_gib=10)" in restore_graph
    assert "recorded_artifacts" in restore_graph
    assert "restored_sha256" in restore_graph
    assert "RESTORED_GRAPH_PATH" in restore_graph
    assert "feature_checkpoint_id" in restore_graph

    assert '"--graph-path"' in train_gnn
    assert "RESTORED_GRAPH_PATH" in train_gnn
    assert '"--checkpoint-path"' in train_gnn
    assert '"--checkpoint-every"' in train_gnn
    assert 'training_arguments.extend(("--resume-from", training_checkpoint_path))' in train_gnn
    assert "training_checkpoint_path.is_file()" in train_gnn
    assert 'drive_processed / "gnn_checkpoints"' in persist_model
    assert 'f".uploading-{gnn_checkpoint_id}"' in persist_model
    assert "model_sha256" in persist_model
    assert "checkpoint_test_evaluated" in persist_model
    assert "training_checkpoint_path" in persist_model
    assert persist_model.index("os.replace(staging_checkpoint, final_checkpoint)") < (
        persist_model.index('write_json_atomic(drive_processed / "_GNN_SUCCESS.json"')
    )
    assert persist_model.index('write_json_atomic(drive_processed / "_GNN_SUCCESS.json"') < (
        persist_model.index("for candidate in gnn_checkpoint_root.iterdir()")
    )


def test_colab_notebook_checks_every_required_source_table() -> None:
    notebook = _notebook()
    code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    for table in ("demo", "drug", "indi", "reac", "outc", "delete"):
        assert f'"{table}"' in code
    assert 'glob("*.sql.gz")' in code
    assert "len(drugcentral_dumps) != 1" in code
    assert "any(missing_coverage.values())" in code
