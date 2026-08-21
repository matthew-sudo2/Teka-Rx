"""Canonical local data paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Resolve the three data lifecycle boundaries from one root."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    def create(self) -> None:
        for path in (self.raw, self.interim, self.processed):
            path.mkdir(parents=True, exist_ok=True)
