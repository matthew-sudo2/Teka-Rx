from pathlib import Path

import pytest

from tekarx.extract.dailymed import DAILYMED_DATASETS
from tekarx.paths import DataPaths


def test_data_paths_create_only_lifecycle_boundaries(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path / "data")
    paths.create()

    assert sorted(item.name for item in paths.root.iterdir()) == ["interim", "processed", "raw"]


def test_dailymed_dataset_registry_is_intentionally_small() -> None:
    assert set(DAILYMED_DATASETS) == {"metadata", "pharmacologic-class", "rxnorm"}


def test_invalid_faers_quarter_fails_before_network(tmp_path: Path) -> None:
    from tekarx.extract.faers import extract_faers

    with pytest.raises(ValueError, match="YYYYQ1"):
        extract_faers(data_dir=tmp_path, quarter="Q1-2024", url="https://example.test")
