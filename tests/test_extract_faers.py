import zipfile
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tekarx.extract.common import DownloadError
from tekarx.extract.faers import (
    _extract_zip_once,
    _parse_faers_archives,
    extract_faers,
    resolve_faers_source,
)


def test_faers_quarter_is_downloaded_and_extracted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("ascii/DEMO24Q1.txt", "primaryid$age\n1$42\n")
        output.writestr("README.txt", "fixture")
    payload = archive.read_bytes()

    @contextmanager
    def fake_stream(*args, **kwargs):
        request = httpx.Request("GET", "https://example.test/faers.zip")
        yield httpx.Response(200, request=request, content=payload)

    monkeypatch.setattr(httpx, "stream", fake_stream)
    extract_faers(
        data_dir=tmp_path / "data",
        quarter="2024q1",
        url="https://example.test/faers.zip",
    )

    extracted = tmp_path / "data/raw/faers/2024Q1/extracted"
    assert (extracted / "ascii/DEMO24Q1.txt").is_file()
    assert (extracted / ".complete").is_file()


def test_unsafe_zip_path_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "no")

    with pytest.raises(DownloadError, match="unsafe path"):
        _extract_zip_once(archive, tmp_path / "output")


def test_quarter_resolves_to_official_ascii_url() -> None:
    quarter, url = resolve_faers_source(quarter="2024q1", url=None)

    assert quarter == "2024Q1"
    assert url == "https://fis.fda.gov/content/Exports/faers_ascii_2024q1.zip"


def test_official_index_parser_finds_available_quarters() -> None:
    html = """
        <a href="/content/Exports/faers_ascii_2026q1.zip">ASCII</a>
        <a href="/content/Exports/faers_xml_2026q1.zip">XML</a>
        <a href="/content/Exports/faers_ascii_2025q4.zip">ASCII</a>
    """

    assert _parse_faers_archives(html, base_url="https://fis.fda.gov/index.html") == {
        "2025Q4": "https://fis.fda.gov/content/Exports/faers_ascii_2025q4.zip",
        "2026Q1": "https://fis.fda.gov/content/Exports/faers_ascii_2026q1.zip",
    }
