"""Extract the complete official DrugCentral database dump."""

from __future__ import annotations

from pathlib import Path

from tekarx.extract.common import DownloadRecord, download_file

DEFAULT_DRUGCENTRAL_URL = "https://unmtid-dbs.net/download/drugcentral.dump.11012023.sql.gz"


def extract_drugcentral(
    *,
    data_dir: Path,
    url: str = DEFAULT_DRUGCENTRAL_URL,
    expected_sha256: str | None = None,
) -> DownloadRecord:
    """Download the PostgreSQL dump without transforming its schema."""
    directory = data_dir / "raw" / "drugcentral"
    return download_file(
        source="DrugCentral",
        dataset="postgres-dump-2023-11-01",
        url=url,
        destination=directory / "drugcentral.dump.11012023.sql.gz",
        manifest_path=directory / "manifest.json",
        expected_sha256=expected_sha256,
    )
