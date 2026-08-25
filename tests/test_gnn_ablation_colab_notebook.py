from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "gnn_ablation_study_colab.ipynb"


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


def test_ablation_notebook_is_clean_and_compilable() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["metadata"]["accelerator"] == "GPU"

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        compile(_source(cell), f"{NOTEBOOK_PATH.name}:{cell.get('id')}", "exec")


def test_ablation_notebook_uses_versioned_full_checkpoints_and_locked_test() -> None:
    notebook = _notebook()
    code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    required = (
        "drive.mount",
        "/content/drive/MyDrive/Teka-Rx-full/data",
        "_GRAPH_SUCCESS.json",
        "_GNN_SUCCESS.json",
        "verified_complete",
        "graph_checkpoint_id",
        "gnn_checkpoint_id",
        "CHECKPOINT_GIT_SHA",
        "git_sha",
        "gnn-full",
        "graph_checkpoints",
        "gnn_checkpoints",
        "xgboost_baseline.json",
        "tekarx_inductive_gnn.pt",
        "split_value=1",
        '"test_evaluated": False',
        "2024Q1",
        "2024Q2",
    )
    for marker in required:
        assert marker in code

    forbidden = (
        "--evaluate-test",
        "split_value=2",
        'split_id == 2]',
        'split_id == 2)',
        "test_scores =",
        "test_labels =",
        "github_pat_",
        "AIza",
    )
    for marker in forbidden:
        assert marker not in code

    restore = _cell_source(notebook, "restore-checkpoints")
    assert "validate_checkpoint(drive_graph_root" in restore
    assert "validate_checkpoint(drive_gnn_root" in restore
    assert "sha256_file" in _cell_source(notebook, "helpers")
    assert "graph_marker" in restore
    assert "gnn_marker" in restore


def test_ablation_notebook_has_matched_retraining_arms_and_restart_checkpoints() -> None:
    notebook = _notebook()
    variants = _cell_source(notebook, "build-variants")
    training = _cell_source(notebook, "train-ablation-arms")

    for arm in (
        "full_gnn",
        "no_normalized_dosage",
        "patient_only",
        "shuffled_topology",
        "xgboost_saved_baseline",
    ):
        prediction_code = _cell_source(notebook, "validation-predictions")
        assert arm in "\n".join((variants, training, prediction_code))

    assert "prospective-no-dosage" in training
    assert "train_inductive_gnn" in training
    assert "evaluate_test=False" in training
    assert "training_checkpoints" in training
    assert "resume_from=checkpoint if checkpoint.is_file() else None" in training
    assert "EPOCHS = 100" in _cell_source(notebook, "configuration")
    assert "PATIENCE = 15" in _cell_source(notebook, "configuration")

    assert "split_offsets" in variants
    assert "Rotating drug endpoints" in variants
    assert '"patient_degree_preserved": True' in variants
    assert '"split_drug_frequency_preserved"' in variants
    assert 'manifest["created_at_utc"] = datetime.now' not in variants


def test_ablation_notebook_reports_threshold_and_probability_metrics() -> None:
    notebook = _notebook()
    configuration = _cell_source(notebook, "configuration")
    metrics = _cell_source(notebook, "metrics-and-thresholds")
    persistence = _cell_source(notebook, "persist-results")

    assert "DECISION_THRESHOLD = 0.50" in configuration
    assert "TARGET_RECALL = 0.95" in configuration
    assert "THRESHOLD_SWEEP" in configuration

    for metric in (
        '"accuracy"',
        '"balanced_accuracy"',
        '"precision"',
        '"recall"',
        '"f1"',
        '"specificity"',
        '"npv"',
        '"mcc"',
        '"auroc"',
        '"auprc"',
        '"brier_score"',
        '"log_loss"',
        '"tp"',
        '"fp"',
        '"tn"',
        '"fn"',
    ):
        assert metric in metrics

    for policy in (
        "FIXED_THRESHOLD_POLICY",
        "validation_max_f1_exploratory",
        "validation_recall_95_exploratory",
    ):
        assert policy in metrics

    for artifact in (
        "validation_predictions.parquet",
        "metrics_by_threshold.parquet",
        "threshold_sweep.parquet",
        "ablation_manifest.json",
        "ablation_summary.md",
    ):
        assert artifact in persistence

    assert 'compression="snappy"' in persistence
    assert '"test_evaluated": False' in persistence
