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
        "build-prospective",
        "--split-preset",
        "--skip-graph",
        "build-graph",
        "--graph-storage",
        "memory-mapped",
        "tekarx_graph_arrays",
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

    assert code.count('"train-gnn"') == 1
    assert code.index('"build-prospective"') < code.index('"train-gnn"')
    assert code.index('"build-graph"') < code.index('"train-gnn"')
    assert code.index("_GRAPH_SUCCESS.json") < code.index('"train-gnn"')


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
