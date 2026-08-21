from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tekarx.extract.common import DownloadError, download_file, sha256_file


def test_download_is_atomic_manifested_and_restartable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    @contextmanager
    def fake_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request("GET", "https://example.test/source.zip")
        response = httpx.Response(200, request=request, content=b"source bytes")
        yield response

    monkeypatch.setattr(httpx, "stream", fake_stream)
    destination = tmp_path / "raw" / "source.zip"
    manifest = tmp_path / "raw" / "manifest.json"
    arguments = {
        "source": "test source",
        "dataset": "test dataset",
        "url": "https://example.test/source.zip",
        "destination": destination,
        "manifest_path": manifest,
    }

    first = download_file(**arguments)
    second = download_file(**arguments)

    assert calls == 1
    assert destination.read_bytes() == b"source bytes"
    assert first.sha256 == sha256_file(destination)
    assert first.cached is False
    assert second.cached is True
    assert manifest.is_file()


def test_existing_unmanifested_file_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "source.zip"
    destination.write_bytes(b"unknown provenance")

    with pytest.raises(DownloadError, match="refusing to overwrite"):
        download_file(
            source="test",
            dataset="test",
            url="https://example.test/source.zip",
            destination=destination,
            manifest_path=tmp_path / "manifest.json",
        )


def test_retryable_response_is_retried_without_leaving_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    @contextmanager
    def fake_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request("GET", "https://example.test/source.zip")
        status = 503 if calls == 1 else 200
        yield httpx.Response(status, request=request, content=b"complete")

    monkeypatch.setattr(httpx, "stream", fake_stream)
    monkeypatch.setattr("tekarx.extract.common.time.sleep", lambda _: None)
    destination = tmp_path / "source.zip"
    record = download_file(
        source="test",
        dataset="retry",
        url="https://example.test/source.zip",
        destination=destination,
        manifest_path=tmp_path / "manifest.json",
    )

    assert calls == 2
    assert record.size_bytes == len(b"complete")
    assert destination.read_bytes() == b"complete"
    assert not destination.with_suffix(".zip.part").exists()
