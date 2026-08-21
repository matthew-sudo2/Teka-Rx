"""Extract compact official DailyMed mapping datasets."""

from __future__ import annotations

from pathlib import Path

from tekarx.extract.common import DownloadRecord, download_file

DAILYMED_DATASETS = {
    "rxnorm": (
        "https://dailymed-data.nlm.nih.gov/public-release-files/rxnorm_mappings.zip",
        "rxnorm_mappings.zip",
    ),
    "pharmacologic-class": (
        "https://dailymed-data.nlm.nih.gov/public-release-files/pharmacologic_class_mappings.zip",
        "pharmacologic_class_mappings.zip",
    ),
    "metadata": (
        "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_zip_files_meta_data.zip",
        "dm_spl_zip_files_meta_data.zip",
    ),
}


def extract_dailymed(
    *,
    data_dir: Path,
    dataset: str,
    url: str | None = None,
    expected_sha256: str | None = None,
) -> DownloadRecord:
    """Download one mapping archive used to select relevant SPL labels."""
    if dataset not in DAILYMED_DATASETS:
        allowed = ", ".join(sorted(DAILYMED_DATASETS))
        raise ValueError(f"unknown DailyMed dataset {dataset!r}; choose: {allowed}")
    default_url, filename = DAILYMED_DATASETS[dataset]
    directory = data_dir / "raw" / "dailymed"
    return download_file(
        source="NLM DailyMed",
        dataset=dataset,
        url=url or default_url,
        destination=directory / filename,
        manifest_path=directory / "manifest.json",
        expected_sha256=expected_sha256,
    )
