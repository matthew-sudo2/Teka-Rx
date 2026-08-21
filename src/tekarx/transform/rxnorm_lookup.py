"""Build an offline DrugCentral/RxNorm name-to-structure lookup."""

from __future__ import annotations

import gzip
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from tekarx.extract.common import RETRYABLE_STATUS_CODES, sha256_file
from tekarx.transform.drug_dictionary import (
    _COPY_RE,
    DrugDictionaryBuildError,
    _drugcentral_aliases,
    _identifier,
    _resolve_components,
    _unescape_copy,
    normalize_drug_name,
)

RXNORM_API = "https://rxnav.nlm.nih.gov/REST"
LOOKUP_COLUMNS = ["rxnorm_name", "struct_id", "rxcui", "source", "query_name"]


@dataclass(frozen=True)
class RxNormLookupRecord:
    """Summary and provenance for one lookup build."""

    output_path: str
    rows: int
    local_rows: int
    api_rows: int
    api_queries: int
    rxnorm_table_found: bool
    rxnorm_identifier_type_found: bool
    compression: str = "snappy"


def build_rxnorm_lookup(
    *,
    data_dir: Path,
    use_api: bool = False,
    batch_size: int = 250,
    requests_per_second: float = 10.0,
    max_names: int | None = None,
) -> RxNormLookupRecord:
    """Build local RxNorm links and optionally resolve unknown FAERS names with RxNav."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if not 0 < requests_per_second <= 20:
        raise ValueError("requests per second must be greater than 0 and no more than 20")
    if max_names is not None and max_names < 1:
        raise ValueError("max names must be positive")

    dump = _one_dump(data_dir)
    structures = data_dir / "interim" / "drugcentral" / "structures.parquet"
    synonyms = data_dir / "interim" / "drugcentral" / "synonyms.parquet"
    for path in (structures, synonyms):
        if not path.is_file():
            raise DrugDictionaryBuildError(f"missing required input: {path}")

    aliases = _drugcentral_aliases(structures, synonyms)
    local_rows, table_found, identifier_type_found = _local_rxnorm_rows(
        dump=dump,
        dailymed=data_dir / "interim" / "dailymed" / "rxnorm.parquet",
    )
    api_rows: list[dict[str, object]] = []
    api_queries = 0
    if use_api:
        unknown_names = _unknown_faers_names(data_dir)
        if max_names is not None:
            unknown_names = unknown_names[:max_names]
        cache = data_dir / "interim" / "rxnorm" / "rxnav_api_cache.jsonl"
        responses, api_queries = _query_rxnav(
            unknown_names,
            cache=cache,
            batch_size=batch_size,
            requests_per_second=requests_per_second,
        )
        api_rows = _api_lookup_rows(responses, aliases=aliases)

    rows = _deduplicate_rows([*local_rows, *api_rows])
    destination = data_dir / "interim" / "drugcentral" / "rxnorm_lookup.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(rows, schema=_lookup_schema())
    pq.write_table(table, temporary, compression="snappy")
    os.replace(temporary, destination)

    record = RxNormLookupRecord(
        output_path=str(destination),
        rows=len(rows),
        local_rows=len(local_rows),
        api_rows=len(api_rows),
        api_queries=api_queries,
        rxnorm_table_found=table_found,
        rxnorm_identifier_type_found=identifier_type_found,
    )
    _write_manifest(
        destination.with_name("rxnorm_lookup_manifest.json"),
        record=record,
        dump=dump,
        use_api=use_api,
    )
    return record


def _local_rxnorm_rows(
    *, dump: Path, dailymed: Path
) -> tuple[list[dict[str, object]], bool, bool]:
    direct_rows, rxnorm_type_ids, table_found = _scan_rxnorm_metadata(dump)
    identifier_type_found = bool(rxnorm_type_ids)
    if not rxnorm_type_ids or not dailymed.is_file():
        return direct_rows, table_found, identifier_type_found

    identifiers = _scan_rxnorm_identifiers(dump, rxnorm_type_ids)
    if not identifiers:
        return direct_rows, table_found, identifier_type_found
    names_by_rxcui: dict[str, set[str]] = defaultdict(set)
    for row in pq.read_table(dailymed, columns=["rxcui", "rxstring"]).to_pylist():
        if row["rxcui"] and row["rxstring"]:
            names_by_rxcui[str(row["rxcui"])].add(str(row["rxstring"]))
    for rxcui, struct_id in identifiers:
        for name in names_by_rxcui.get(rxcui, set()):
            direct_rows.append(_lookup_row(name, struct_id, rxcui, "drugcentral_identifier"))
    return direct_rows, table_found, identifier_type_found


def _scan_rxnorm_metadata(dump: Path) -> tuple[list[dict[str, object]], set[str], bool]:
    rows: list[dict[str, object]] = []
    rxnorm_type_ids: set[str] = set()
    current: dict[str, object] | None = None
    table_found = False
    with gzip.open(dump, "rt", encoding="utf-8", errors="replace", newline="") as stream:
        for line in tqdm(stream, desc="Checking DrugCentral for RxNorm", unit="lines"):
            stripped = line.rstrip("\r\n")
            if current is None:
                match = _COPY_RE.match(stripped)
                if match is None:
                    continue
                table = match.group("table").lower()
                columns = [_identifier(item) for item in match.group("columns").split(",")]
                if table == "id_type":
                    current = {"kind": "id_type", "columns": columns}
                elif "rxnorm" in table:
                    table_found = True
                    current = {"kind": "direct", "columns": columns}
                continue
            if stripped == r"\.":
                current = None
                continue
            values = [_copy_value(item) for item in stripped.split("\t")]
            columns = current["columns"]
            assert isinstance(columns, list)
            row = dict(zip(columns, values, strict=True))
            if current["kind"] == "id_type":
                description = " ".join(str(row.get(key) or "") for key in ("type", "description"))
                if (
                    "RXNORM" in description.upper() or "RXCUI" in description.upper()
                ) and row.get("id") is not None:
                    rxnorm_type_ids.add(str(row["id"]))
            else:
                parsed = _direct_rxnorm_row(row)
                if parsed is not None:
                    rows.append(parsed)
    return rows, rxnorm_type_ids, table_found


def _scan_rxnorm_identifiers(dump: Path, id_types: set[str]) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    columns: list[str] | None = None
    with gzip.open(dump, "rt", encoding="utf-8", errors="replace", newline="") as stream:
        for line in stream:
            stripped = line.rstrip("\r\n")
            if columns is None:
                match = _COPY_RE.match(stripped)
                if match is not None and match.group("table").lower() == "identifier":
                    columns = [_identifier(item) for item in match.group("columns").split(",")]
                continue
            if stripped == r"\.":
                break
            values = [_copy_value(item) for item in stripped.split("\t")]
            row = dict(zip(columns, values, strict=True))
            if str(row.get("id_type")) not in id_types:
                continue
            try:
                results.append((str(row["identifier"]), int(str(row["struct_id"]))))
            except (KeyError, TypeError, ValueError):
                continue
    return results


def _direct_rxnorm_row(row: dict[str, str | None]) -> dict[str, object] | None:
    name = _first(row, "rxnorm_name", "rxstring", "name", "str")
    struct_id = _first(row, "struct_id", "structure_id")
    if not name or not struct_id:
        return None
    try:
        return _lookup_row(
            name,
            int(struct_id),
            _first(row, "rxcui", "rxnorm_id", "identifier"),
            "drugcentral_rxnorm_table",
        )
    except ValueError:
        return None


def _unknown_faers_names(data_dir: Path) -> list[str]:
    dictionary = data_dir / "processed" / "drug_dictionary.parquet"
    if not dictionary.is_file():
        raise DrugDictionaryBuildError(
            "build drug_dictionary.parquet once before using --use-api so unknown names are known"
        )
    table = pq.read_table(dictionary, columns=["faers_raw", "dc_id"])
    unknown = {
        str(name)
        for name, dc_id in zip(
            table["faers_raw"].to_pylist(),
            table["dc_id"].to_pylist(),
            strict=True,
        )
        if name and dc_id == 0
    }
    return sorted(unknown)


def _query_rxnav(
    names: list[str], *, cache: Path, batch_size: int, requests_per_second: float
) -> tuple[dict[str, list[dict[str, str]]], int]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    responses = _read_api_cache(cache)
    pending = [name for name in names if name not in responses]
    rate_limiter = _RateLimiter(requests_per_second)
    queried = 0
    with httpx.Client(
        base_url=RXNORM_API,
        follow_redirects=True,
        timeout=30,
        headers={"User-Agent": "TekaRx/0.1 research data pipeline"},
    ) as client, ThreadPoolExecutor(max_workers=20, thread_name_prefix="rxnorm") as executor:
        for offset in tqdm(range(0, len(pending), batch_size), desc="RxNorm API batches"):
            batch = pending[offset : offset + batch_size]
            futures = {
                executor.submit(_request_candidates, client, name, rate_limiter): name
                for name in batch
            }
            checkpoint: list[dict[str, object]] = []
            for future in as_completed(futures):
                name = futures[future]
                candidates = future.result()
                responses[name] = candidates
                checkpoint.append({"query_name": name, "candidates": candidates})
                queried += 1
                if len(checkpoint) >= 25:
                    _append_api_cache(cache, checkpoint)
                    checkpoint.clear()
            if checkpoint:
                _append_api_cache(cache, checkpoint)
    return responses, queried


def _request_candidates(
    client: httpx.Client, name: str, rate_limiter: _RateLimiter
) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            rate_limiter.wait()
            response = client.get(
                "/approximateTerm.json",
                params={"term": name, "maxEntries": 5},
            )
            response.raise_for_status()
            candidates = response.json().get("approximateGroup", {}).get("candidate", [])
            best_rank = min((int(item.get("rank", 999)) for item in candidates), default=999)
            return [
                {
                    "rxcui": str(item.get("rxcui", "")),
                    "name": str(item.get("name", "")),
                }
                for item in candidates
                if int(item.get("rank", 999)) == best_rank and item.get("name")
            ]
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, TypeError) as exc:
            last_error = exc
            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                exc.response.status_code in RETRYABLE_STATUS_CODES
            )
            if not retryable:
                raise
            if attempt < 3:
                time.sleep(min(2**attempt, 8))
    raise DrugDictionaryBuildError(f"RxNorm API failed for {name!r}") from last_error


class _RateLimiter:
    """Thread-safe fixed-interval limiter that also covers retry requests."""

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / requests_per_second
        self._next_request = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request - now)
            if delay:
                time.sleep(delay)
            self._next_request = max(self._next_request, time.monotonic()) + self._interval


def _api_lookup_rows(
    responses: dict[str, list[dict[str, str]]], *, aliases: dict[str, object]
) -> list[dict[str, object]]:
    canonical = aliases["canonical"]
    synonyms = aliases["synonyms"]
    assert isinstance(canonical, dict)
    assert isinstance(synonyms, dict)

    def resolve(part: str) -> int:
        canonical_ids = canonical.get(part, set())
        synonym_ids = synonyms.get(part, set())
        ids = canonical_ids if canonical_ids else synonym_ids
        return next(iter(ids)) if len(ids) == 1 else 0

    rows: list[dict[str, object]] = []
    for query_name, candidates in responses.items():
        candidate_links: list[tuple[dict[str, str], tuple[int, ...]]] = []
        for candidate in candidates:
            ids = _resolve_components(candidate["name"], resolve)
            if ids:
                candidate_links.append((candidate, ids))
        distinct = {ids for _, ids in candidate_links}
        if len(distinct) != 1:
            continue
        selected_ids = next(iter(distinct))
        selected = next(candidate for candidate, ids in candidate_links if ids == selected_ids)
        for struct_id in selected_ids:
            rows.append(
                _lookup_row(
                    selected["name"],
                    struct_id,
                    selected.get("rxcui"),
                    "rxnorm_api",
                    query_name=query_name,
                )
            )
    return rows


def _read_api_cache(path: Path) -> dict[str, list[dict[str, str]]]:
    records: dict[str, list[dict[str, str]]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                records[str(row["query_name"])] = list(row["candidates"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise DrugDictionaryBuildError(
                    f"invalid RxNorm cache line {number}: {path}"
                ) from exc
    return records


def _append_api_cache(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _lookup_row(
    name: str,
    struct_id: int,
    rxcui: str | None,
    source: str,
    *,
    query_name: str | None = None,
) -> dict[str, object]:
    return {
        "rxnorm_name": name,
        "struct_id": struct_id,
        "rxcui": rxcui,
        "source": source,
        "query_name": query_name,
    }


def _deduplicate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique = {
        tuple(row[column] for column in LOOKUP_COLUMNS): row
        for row in rows
        if normalize_drug_name(str(row["rxnorm_name"]))
    }
    return sorted(
        unique.values(),
        key=lambda row: (
            str(row["query_name"] or ""),
            str(row["rxnorm_name"]),
            int(row["struct_id"]),
        ),
    )


def _lookup_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("rxnorm_name", pa.string(), nullable=False),
            pa.field("struct_id", pa.int64(), nullable=False),
            pa.field("rxcui", pa.string()),
            pa.field("source", pa.string(), nullable=False),
            pa.field("query_name", pa.string()),
        ]
    )


def _one_dump(data_dir: Path) -> Path:
    dumps = sorted((data_dir / "raw" / "drugcentral").glob("*.sql.gz"))
    if len(dumps) != 1:
        raise DrugDictionaryBuildError(
            f"expected one DrugCentral SQL dump, found {len(dumps)}"
        )
    return dumps[0]


def _copy_value(value: str) -> str | None:
    return None if value == r"\N" else _unescape_copy(value)


def _first(row: dict[str, str | None], *names: str) -> str | None:
    return next((row[name] for name in names if row.get(name)), None)


def _write_manifest(
    path: Path, *, record: RxNormLookupRecord, dump: Path, use_api: bool
) -> None:
    payload = {
        "dataset": "DrugCentral/RxNorm name-to-structure lookup",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "drugcentral_dump": str(dump),
        "drugcentral_dump_sha256": sha256_file(dump),
        "rxnorm_api": RXNORM_API if use_api else None,
        "rxnorm_api_terms": "https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html",
        "record": asdict(record),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
