"""Shared restart, checksum, and provenance behavior for source downloads."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tqdm.auto import tqdm

CHUNK_SIZE = 1024 * 1024
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class DownloadRecord:
    """Provenance for one immutable source artifact."""

    source: str
    dataset: str
    url: str
    retrieved_at_utc: str
    path: str
    size_bytes: int
    sha256: str
    cached: bool = False


class DownloadError(RuntimeError):
    """Raised when an immutable download cannot be completed safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    *,
    source: str,
    dataset: str,
    url: str,
    destination: Path,
    manifest_path: Path,
    expected_sha256: str | None = None,
    timeout_seconds: float = 120,
    max_retries: int = 3,
) -> DownloadRecord:
    """Download atomically, or reuse a manifest-verified immutable artifact."""
    manifest = _read_manifest(manifest_path, source)
    cached = _cached_record(manifest, dataset, url, destination)
    if cached is not None:
        if expected_sha256 and cached.sha256.lower() != expected_sha256.lower():
            raise DownloadError("cached artifact does not match --sha256")
        return cached
    if destination.exists():
        raise DownloadError(
            f"refusing to overwrite unverified existing file: {destination}. "
            "Remove it manually after checking provenance."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        checksum = _download_with_retry(
            url=url,
            dataset=dataset,
            temporary=temporary,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        if expected_sha256 and checksum.lower() != expected_sha256.lower():
            raise DownloadError(
                f"checksum mismatch for {dataset}: expected {expected_sha256}, got {checksum}"
            )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    record = DownloadRecord(
        source=source,
        dataset=dataset,
        url=url,
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        path=str(destination),
        size_bytes=destination.stat().st_size,
        sha256=checksum,
    )
    _upsert_manifest(manifest_path, manifest, record)
    return record


def _download_with_retry(
    *, url: str, dataset: str, temporary: Path, timeout_seconds: float, max_retries: int
) -> str:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        try:
            with httpx.stream(
                "GET",
                url,
                follow_redirects=True,
                timeout=timeout_seconds,
                headers={"User-Agent": "TekaRx/0.1 research data pipeline"},
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                total = int(content_length) if content_length and content_length.isdigit() else None
                with (
                    temporary.open("xb") as output,
                    tqdm(
                        total=total,
                        desc=f"Downloading {dataset}",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress,
                ):
                    for chunk in response.iter_bytes(CHUNK_SIZE):
                        if chunk:
                            output.write(chunk)
                            digest.update(chunk)
                            progress.update(len(chunk))
            return digest.hexdigest()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                raise
            last_error = exc
        except httpx.RequestError as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(min(2**attempt, 8))
    raise DownloadError(f"download failed after {max_retries + 1} attempts: {url}") from last_error


def _read_manifest(path: Path, source: str) -> dict[str, object]:
    if not path.exists():
        return {"source": source, "artifacts": []}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownloadError(f"invalid manifest: {path}") from exc
    if manifest.get("source") != source or not isinstance(manifest.get("artifacts"), list):
        raise DownloadError(f"invalid manifest structure: {path}")
    return manifest


def _cached_record(
    manifest: dict[str, object], dataset: str, url: str, destination: Path
) -> DownloadRecord | None:
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    entry = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict)
            and item.get("dataset") == dataset
            and item.get("url") == url
            and item.get("path") == str(destination)
        ),
        None,
    )
    if entry is None or not destination.is_file():
        return None
    if sha256_file(destination) != entry.get("sha256"):
        raise DownloadError(f"cached artifact checksum mismatch: {destination}")
    normalized = dict(entry)
    normalized["cached"] = True
    return DownloadRecord(**normalized)


def _upsert_manifest(path: Path, manifest: dict[str, object], record: DownloadRecord) -> None:
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    artifacts[:] = [
        item
        for item in artifacts
        if not isinstance(item, dict) or item.get("dataset") != record.dataset
    ]
    artifacts.append(asdict(record))
    artifacts.sort(key=lambda item: str(item.get("dataset")) if isinstance(item, dict) else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
