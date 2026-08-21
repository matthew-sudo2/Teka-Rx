import json
from pathlib import Path

from tekarx.studies import preset_quarters, write_split_plan


def test_gnn_small_preset_has_chronological_nonoverlapping_splits() -> None:
    quarters = preset_quarters("gnn-small")

    assert quarters == ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2"]
    assert len(quarters) == len(set(quarters))


def test_gnn_full_preset_has_five_training_years_and_fixed_holdouts() -> None:
    quarters = preset_quarters("gnn-full")

    assert quarters[:4] == ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]
    assert quarters[-6:] == [
        "2023Q1",
        "2023Q2",
        "2023Q3",
        "2023Q4",
        "2024Q1",
        "2024Q2",
    ]
    assert len(quarters) == 22
    assert len(quarters) == len(set(quarters))


def test_split_plan_records_leakage_keys(tmp_path: Path) -> None:
    destination = write_split_plan(data_dir=tmp_path / "data", name="gnn-small")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["group_key"] == "CASEID"
    assert payload["version_key"] == "CASEVERSION"
    assert payload["splits"]["validation"] == ["2024Q1"]
