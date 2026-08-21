"""Extract an official FAERS quarterly ASCII archive."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import httpx

from tekarx.extract.common import DownloadError, DownloadRecord, download_file

_QUARTER = re.compile(r"^20\d{2}Q[1-4]$")
_QUARTER_IN_URL = re.compile(r"(?i)faers_ascii_(20\d{2}q[1-4])\.zip")
_ARCHIVE_LINK = re.compile(r"""(?i)href=["']([^"']*faers_ascii_(20\d{2}q[1-4])\.zip)["']""")
FAERS_INDEX_URL = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
FAERS_ARCHIVE_TEMPLATE = "https://fis.fda.gov/content/Exports/faers_ascii_{quarter}.zip"


def extract_faers(
    *,
    data_dir: Path,
    quarter: str | None = None,
    url: str | None = None,
    expected_sha256: str | None = None,
) -> DownloadRecord:
    """Download and safely expand one non-cumulative FAERS ASCII quarter."""
    normalized_quarter, resolved_url = resolve_faers_source(quarter=quarter, url=url)
    quarter_dir = data_dir / "raw" / "faers" / normalized_quarter
    archive = quarter_dir / "source.zip"
    record = download_file(
        source="FDA FAERS/AEMS quarterly ASCII data",
        dataset=normalized_quarter,
        url=resolved_url,
        destination=archive,
        manifest_path=data_dir / "raw" / "faers" / "manifest.json",
        expected_sha256=expected_sha256,
    )
    _extract_zip_once(archive, quarter_dir / "extracted")
    return record


def resolve_faers_source(*, quarter: str | None, url: str | None) -> tuple[str, str]:
    """Resolve convenient CLI input to an explicit quarter and official archive URL."""
    normalized_quarter = quarter.upper() if quarter else None
    if normalized_quarter and not _QUARTER.fullmatch(normalized_quarter):
        raise ValueError("quarter must use YYYYQ1 through YYYYQ4")

    if url:
        if normalized_quarter is None:
            match = _QUARTER_IN_URL.search(url)
            if match is None:
                raise ValueError("--quarter is required when it cannot be inferred from --url")
            normalized_quarter = match.group(1).upper()
        return normalized_quarter, url

    if normalized_quarter:
        return normalized_quarter, FAERS_ARCHIVE_TEMPLATE.format(quarter=normalized_quarter.lower())

    archives = discover_faers_archives()
    if not archives:
        raise DownloadError("the FDA index did not contain any FAERS ASCII archives")
    latest = max(archives)
    return latest, archives[latest]


def discover_faers_archives() -> dict[str, str]:
    """Read the official index and return available modern FAERS ASCII quarters."""
    response = httpx.get(
        FAERS_INDEX_URL,
        follow_redirects=True,
        timeout=30,
        headers={"User-Agent": "TekaRx/0.1 research data pipeline"},
    )
    response.raise_for_status()
    return _parse_faers_archives(response.text, base_url=str(response.url))


def _parse_faers_archives(html: str, *, base_url: str) -> dict[str, str]:
    return {
        quarter.upper(): urljoin(base_url, href) for href, quarter in _ARCHIVE_LINK.findall(html)
    }


def _extract_zip_once(archive: Path, destination: Path) -> None:
    marker = destination / ".complete"
    if marker.is_file():
        return
    if destination.exists() and any(destination.iterdir()):
        raise DownloadError(f"refusing to replace incomplete extraction: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = (destination / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise DownloadError(f"unsafe path in FAERS archive: {member.filename}")
            source.extractall(destination)
        marker.write_text("verified extraction\n", encoding="utf-8")
    except Exception:
        marker.unlink(missing_ok=True)
        raise
