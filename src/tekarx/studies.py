"""Source-controlled dataset splits for reproducible experiments."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

FAERS_PRESETS: dict[str, dict[str, list[str]]] = {
    "gnn-small": {
        "train": ["2023Q1", "2023Q2", "2023Q3", "2023Q4"],
        "validation": ["2024Q1"],
        "test": ["2024Q2"],
    },
    "gnn-full": {
        "train": [f"{year}Q{quarter}" for year in range(2019, 2024) for quarter in range(1, 5)],
        "validation": ["2024Q1"],
        "test": ["2024Q2"],
    },
}


def preset_quarters(name: str) -> list[str]:
    """Return unique quarters in chronological split order."""
    try:
        split = FAERS_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(FAERS_PRESETS))
        raise ValueError(f"unknown preset {name!r}; choose: {choices}") from exc
    return list(dict.fromkeys(quarter for values in split.values() for quarter in values))


def write_split_plan(*, data_dir: Path, name: str) -> Path:
    """Persist the exact split beside processed data without changing its definition."""
    if name not in FAERS_PRESETS:
        preset_quarters(name)
    destination = data_dir / "processed" / "splits" / f"faers_{name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "FAERS",
        "preset": name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "group_key": "CASEID",
        "version_key": "CASEVERSION",
        "split_strategy": "quarter-based temporal holdout",
        "splits": FAERS_PRESETS[name],
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination
