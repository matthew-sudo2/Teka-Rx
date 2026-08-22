"""Build a FAERS-to-DrugCentral dictionary with ROR and boxed-warning flags."""

from __future__ import annotations

import gzip
import json
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm

from tekarx.extract.common import sha256_file
from tekarx.transform.drugcentral import _COPY_RE, _identifier, _unescape_copy
from tekarx.transform.duckdb_runtime import configure_duckdb

DRUG_DICTIONARY_BUILD_VERSION = 4
BOXED_WARNING_LOINC = "34066-1"
_BATCH_ROWS = 50_000


class DrugDictionaryBuildError(RuntimeError):
    """Raised when a drug dictionary cannot be built or audited safely."""


@dataclass(frozen=True)
class DrugDictionaryBuildRecord:
    """Summary and provenance for one dictionary build."""

    output_path: str
    rows: int
    exact_matches: int
    fuzzy_matches: int
    unmapped: int
    exact_hit_rate: float
    fuzzy_enabled: bool
    boxed_warning_drugs: int
    compression: str = "snappy"
    cached: bool = False


def build_drug_dictionary(
    *,
    data_dir: Path,
    fuzzy_trigger_rate: float = 0.50,
    fuzzy_score_cutoff: float = 97.0,
    fuzzy_margin: float = 3.0,
    memory_limit: str = "4GB",
    threads: int | None = None,
) -> DrugDictionaryBuildRecord:
    """Create one raw-name linkage row with ingredient-level ROR per FAERS drug."""
    _validate_thresholds(fuzzy_trigger_rate, fuzzy_score_cutoff, fuzzy_margin)
    paths = _required_paths(data_dir)
    source_checksums = {str(path): sha256_file(path) for path in paths.values()}
    output = data_dir / "processed" / "drug_dictionary.parquet"
    manifest = data_dir / "processed" / "drug_dictionary_manifest.json"
    cached = _cached_record(
        manifest=manifest,
        output=output,
        source_checksums=source_checksums,
        fuzzy_trigger_rate=fuzzy_trigger_rate,
        fuzzy_score_cutoff=fuzzy_score_cutoff,
        fuzzy_margin=fuzzy_margin,
    )
    if cached is not None:
        print("Drug dictionary: verified cached artifact", flush=True)
        return cached

    print("Drug dictionary [1/6]: extracting DrugCentral boxed-warning flags", flush=True)
    boxed_warning_path = _build_boxed_warning_flags(
        data_dir=data_dir,
        source=paths["drugcentral_dump"],
        source_sha256=source_checksums[str(paths["drugcentral_dump"])],
        memory_limit=memory_limit,
        threads=threads,
    )
    print("Drug dictionary [2/6]: loading DrugCentral/RxNorm aliases", flush=True)
    aliases = _drugcentral_aliases(paths["structures"], paths["synonyms"])
    rxnorm_aliases = _rxnorm_aliases(paths.get("rxnorm_lookup"))
    atc_by_id = _atc_codes(paths["struct2atc"])
    boxed_ids = {
        int(value)
        for value in pq.read_table(boxed_warning_path, columns=["dc_id"])["dc_id"].to_pylist()
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    database = data_dir / "interim" / f".drug-dictionary-{os.getpid()}.duckdb"
    database.unlink(missing_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(database))
        configure_duckdb(
            connection,
            data_dir=data_dir,
            stage="drug-dictionary",
            memory_limit=memory_limit,
            threads=threads,
        )
        print("Drug dictionary [3/6]: deduplicating retained FAERS exposures", flush=True)
        edges = _sql_literal(paths["drug_edges"].as_posix())
        cohort = _sql_literal(paths["cohort"].as_posix())
        connection.execute(
            f"""
            CREATE TABLE exposures AS
            SELECT DISTINCT e.primaryid,
                   trim(e.drugname) AS faers_raw,
                   nullif(trim(e.prod_ai), '') AS prod_ai
            FROM read_parquet('{edges}') e
            JOIN read_parquet('{cohort}') c USING (primaryid)
            WHERE e.drugname IS NOT NULL AND trim(e.drugname) <> ''
            """
        )
        raw_to_prod_ai: dict[str, list[str]] = defaultdict(list)
        for faers_raw, prod_ai in connection.execute(
            "SELECT DISTINCT faers_raw, prod_ai FROM exposures ORDER BY faers_raw, prod_ai"
        ).fetchall():
            if prod_ai is not None:
                raw_to_prod_ai[faers_raw].append(prod_ai)
            else:
                raw_to_prod_ai.setdefault(faers_raw, [])
        raw_names = list(raw_to_prod_ai)
        print(f"Drug dictionary [4/6]: mapping {len(raw_names):,} unique FAERS names", flush=True)
        mappings, stats = _map_names(
            raw_to_prod_ai,
            aliases=aliases,
            rxnorm_aliases=rxnorm_aliases,
            fuzzy_trigger_rate=fuzzy_trigger_rate,
            fuzzy_score_cutoff=fuzzy_score_cutoff,
            fuzzy_margin=fuzzy_margin,
        )
        linkage_rows = [
            (raw_name, dc_id)
            for raw_name in raw_names
            for dc_id in mappings[raw_name]
        ]
        mapping_table = pa.table(
            {
                "faers_raw": pa.array([row[0] for row in linkage_rows], type=pa.string()),
                "dc_id": pa.array([row[1] for row in linkage_rows], type=pa.int64()),
                "atc_code": pa.array(
                    [atc_by_id.get(row[1]) for row in linkage_rows], type=pa.string()
                ),
                "has_boxed_warning": pa.array(
                    [int(row[1] in boxed_ids) for row in linkage_rows], type=pa.int8()
                ),
            }
        )
        connection.register("drug_mapping", mapping_table)
        print("Drug dictionary [5/6]: calculating ingredient-level ROR", flush=True)
        connection.execute(
            """
            CREATE TABLE mapped_exposures AS
            SELECT DISTINCT e.primaryid, m.dc_id
            FROM exposures e
            JOIN drug_mapping m USING (faers_raw)
            """
        )
        connection.execute(
            f"""
            CREATE TABLE ror_by_dc_id AS
            WITH totals AS (
                SELECT count(*) FILTER (WHERE is_serious = 1) AS total_serious,
                       count(*) FILTER (WHERE is_serious = 0) AS total_nonserious
                FROM read_parquet('{cohort}')
            ), counts AS (
                SELECT e.dc_id,
                       count(*) FILTER (WHERE c.is_serious = 1) AS a,
                       count(*) FILTER (WHERE c.is_serious = 0) AS b
                FROM mapped_exposures e
                JOIN read_parquet('{cohort}') c USING (primaryid)
                GROUP BY e.dc_id
            ), cells AS (
                SELECT dc_id, a, b,
                       total_serious - a AS c,
                       total_nonserious - b AS d
                FROM counts CROSS JOIN totals
            )
            SELECT dc_id,
                   CASE
                       WHEN a = 0 OR b = 0 OR c = 0 OR d = 0
                           THEN ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
                       ELSE (a::DOUBLE * d::DOUBLE) / (b::DOUBLE * c::DOUBLE)
                   END::DOUBLE AS ror
            FROM cells
            """
        )
        destination = _sql_literal(temporary.as_posix())
        print("Drug dictionary [6/6]: writing and auditing Snappy Parquet", flush=True)
        connection.execute(
            f"""
            COPY (
                SELECT m.faers_raw, m.dc_id, m.atc_code, r.ror,
                       m.has_boxed_warning::INTEGER AS has_boxed_warning
                FROM drug_mapping m
                JOIN ror_by_dc_id r USING (dc_id)
                ORDER BY m.faers_raw
            ) TO '{destination}'
            (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000)
            """
        )
        rows = len(linkage_rows)
        connection.close()
        connection = None
        _verify_dictionary(temporary, expected_rows=rows)
        os.replace(temporary, output)
        record = DrugDictionaryBuildRecord(
            output_path=str(output),
            rows=rows,
            exact_matches=stats["exact_matches"],
            fuzzy_matches=stats["fuzzy_matches"],
            unmapped=stats["unmapped"],
            exact_hit_rate=stats["exact_hit_rate"],
            fuzzy_enabled=stats["fuzzy_enabled"],
            boxed_warning_drugs=sum(row[1] in boxed_ids for row in linkage_rows),
        )
        _write_manifest(
            manifest,
            record=record,
            source_checksums=source_checksums,
            fuzzy_trigger_rate=fuzzy_trigger_rate,
            fuzzy_score_cutoff=fuzzy_score_cutoff,
            fuzzy_margin=fuzzy_margin,
        )
        return record
    except BaseException:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        database.unlink(missing_ok=True)


def normalize_drug_name(value: str | None) -> str:
    """Normalize a FAERS or reference name for conservative name matching."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).upper().strip()
    normalized = re.sub(r"[\u00ae\u2122]", "", normalized)
    normalized = re.sub(r"\[[^]]*]", " ", normalized)
    normalized = re.sub(
        r"\s*[\[(](?:CAPSULES?|TABLETS?|INJECTIONS?|SOLUTIONS?|SUSPENSIONS?|CREAMS?|"
        r"OINTMENTS?|PATCH(?:ES)?|SPRAYS?|SYRUPS?|POWDERS?|GELS?)[\])]\s*$",
        "",
        normalized,
    )
    dosage = r"(?:\d+(?:\.\d+)?|\.\d+)\s*(?:MCG|UG|MG|G|KG|ML|L|IU|UNIT(?:S)?|MEQ|MMOL|%)"
    normalized = re.sub(rf"^\s*{dosage}\s+", "", normalized)
    normalized = re.sub(r"^\s*\d{1,4}[.)]?\s+(?=[A-Z])", "", normalized)
    for _ in range(2):
        normalized = re.sub(
            rf"(?:\s+|\s*[-,/]\s*){dosage}(?:\s*/\s*{dosage})?\s*$", "", normalized
        )
        normalized = re.sub(
            r"\s+(?:ER|XR|CR|SR|DR|IR|XL|LA|SA|EXTENDED RELEASE|SUSTAINED RELEASE|"
            r"CONTROLLED RELEASE|DELAYED RELEASE)\s*$",
            "",
            normalized,
        )
    normalized = re.sub(
        r"\b(?:HYDROCHLORIDE|HCL|SODIUM|POTASSIUM|SULFATE|SULPHATE|MALEATE|"
        r"FUMARATE)\b",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"\b(?:CAPSULES?|TABLETS?|INJECTIONS?|INJECTABLE|SOLUTIONS?|SUSPENSIONS?|"
        r"ORAL|TOPICAL|OPHTHALMIC|OTIC|NASAL|RECTAL|VAGINAL|SUBLINGUAL)\b",
        " ",
        normalized,
    )
    normalized = re.sub(rf"\b{dosage}\b", " ", normalized)
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _required_paths(data_dir: Path) -> dict[str, Path]:
    paths = {
        "cohort": data_dir / "processed" / "tekarx_cohort.parquet",
        "drug_edges": data_dir / "processed" / "edges" / "report_drug.parquet",
        "structures": data_dir / "interim" / "drugcentral" / "structures.parquet",
        "synonyms": data_dir / "interim" / "drugcentral" / "synonyms.parquet",
        "struct2atc": data_dir / "interim" / "drugcentral" / "struct2atc.parquet",
    }
    dumps = sorted((data_dir / "raw" / "drugcentral").glob("*.sql.gz"))
    if len(dumps) != 1:
        raise DrugDictionaryBuildError(
            f"expected one raw DrugCentral SQL dump, found {len(dumps)}; "
            "boxed-warning flags require its DailyMed-derived label sections"
        )
    paths["drugcentral_dump"] = dumps[0]
    rxnorm_lookup = data_dir / "interim" / "drugcentral" / "rxnorm_lookup.parquet"
    if rxnorm_lookup.is_file():
        paths["rxnorm_lookup"] = rxnorm_lookup
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise DrugDictionaryBuildError(f"missing required inputs: {missing}")
    return paths


def _drugcentral_aliases(structures_path: Path, synonyms_path: Path) -> dict[str, object]:
    canonical: dict[str, set[int]] = defaultdict(set)
    synonyms: dict[str, set[int]] = defaultdict(set)
    structures = pq.read_table(structures_path, columns=["id", "name"]).to_pylist()
    synonym_rows = pq.read_table(synonyms_path, columns=["id", "name"]).to_pylist()
    for row in structures:
        name = normalize_drug_name(row["name"])
        if name and row["id"] is not None:
            canonical[name].add(int(row["id"]))
    for row in synonym_rows:
        name = normalize_drug_name(row["name"])
        if name and row["id"] is not None:
            synonyms[name].add(int(row["id"]))
    return {"canonical": canonical, "synonyms": synonyms}


def _atc_codes(path: Path) -> dict[int, str]:
    codes: dict[int, set[str]] = defaultdict(set)
    for row in pq.read_table(path, columns=["struct_id", "atc_code"]).to_pylist():
        if row["struct_id"] is not None and row["atc_code"]:
            codes[int(row["struct_id"])].add(str(row["atc_code"]).strip().upper())
    return {dc_id: "|".join(sorted(values)) for dc_id, values in codes.items()}


def _rxnorm_aliases(path: Path | None) -> dict[str, set[int]]:
    """Load optional RxNorm query/canonical-name aliases keyed to DrugCentral IDs."""
    aliases: dict[str, set[int]] = defaultdict(set)
    if path is None:
        return aliases
    table = pq.read_table(path)
    required = {"rxnorm_name", "struct_id"}
    if not required.issubset(table.column_names):
        raise DrugDictionaryBuildError(
            f"invalid RxNorm lookup schema; missing {sorted(required - set(table.column_names))}"
        )
    for row in table.to_pylist():
        if row["struct_id"] is None:
            continue
        dc_id = int(row["struct_id"])
        for column in ("rxnorm_name", "query_name"):
            normalized = normalize_drug_name(row.get(column))
            if normalized:
                aliases[normalized].add(dc_id)
    return aliases


def _map_names(
    raw_to_prod_ai: dict[str, list[str]],
    *,
    aliases: dict[str, object],
    rxnorm_aliases: dict[str, set[int]] | None = None,
    fuzzy_trigger_rate: float,
    fuzzy_score_cutoff: float,
    fuzzy_margin: float,
) -> tuple[dict[str, tuple[int, ...]], dict[str, int | float | bool]]:
    canonical = aliases["canonical"]
    synonyms = aliases["synonyms"]
    assert isinstance(canonical, dict)
    assert isinstance(synonyms, dict)
    rxnorm_aliases = rxnorm_aliases or {}
    mappings: dict[str, tuple[int, ...]] = {}
    exact = 0
    raw_names = list(raw_to_prod_ai)
    conflicts: set[str] = set()

    def exact_dc(value: str) -> tuple[int, ...]:
        return _resolve_components(
            value, lambda part: _exact_alias_id(part, canonical, synonyms)
        )

    def exact_rxnorm(value: str) -> tuple[int, ...]:
        whole_ids = rxnorm_aliases.get(normalize_drug_name(value), set())
        if whole_ids:
            return tuple(sorted(whole_ids))
        return _resolve_components(value, lambda part: _single_id(rxnorm_aliases.get(part, set())))

    for raw_name in raw_names:
        prod_matches = {
            match for value in raw_to_prod_ai[raw_name] if (match := exact_dc(value))
        }
        if len(prod_matches) > 1:
            conflicts.add(raw_name)
            match = ()
        elif prod_matches:
            match = next(iter(prod_matches))
        else:
            match = exact_dc(raw_name)
            if not match:
                rx_matches = {
                    candidate
                    for value in raw_to_prod_ai[raw_name]
                    if (candidate := exact_rxnorm(value))
                }
                if len(rx_matches) > 1:
                    conflicts.add(raw_name)
                elif rx_matches:
                    match = next(iter(rx_matches))
                else:
                    match = exact_rxnorm(raw_name)
        mappings[raw_name] = match or (0,)
        exact += int(match != ())
    exact_rate = exact / len(raw_names) if raw_names else 0.0
    fuzzy_enabled = bool(raw_names) and exact_rate < fuzzy_trigger_rate
    fuzzy_matches = 0
    if fuzzy_enabled:
        searchable: dict[str, int] = {}
        all_aliases = set(canonical) | set(synonyms) | set(rxnorm_aliases)
        for alias in all_aliases:
            ids = (
                set(canonical.get(alias, set()))
                | set(synonyms.get(alias, set()))
                | set(rxnorm_aliases.get(alias, set()))
            )
            if len(ids) == 1:
                searchable[alias] = next(iter(ids))
        choices_by_initial: dict[str, list[str]] = defaultdict(list)
        for choice in searchable:
            choices_by_initial[choice[0]].append(choice)
        fuzzy_by_normalized: dict[str, int] = {}

        def fuzzy_id(normalized: str) -> int:
            if not normalized:
                return 0
            if normalized in fuzzy_by_normalized:
                return fuzzy_by_normalized[normalized]
            choices = choices_by_initial.get(normalized[0], [])
            if not choices:
                fuzzy_by_normalized[normalized] = 0
                return 0
            candidates = process.extract(
                normalized,
                choices,
                scorer=fuzz.WRatio,
                score_cutoff=fuzzy_score_cutoff,
                limit=2,
            )
            if not candidates:
                fuzzy_by_normalized[normalized] = 0
                return 0
            best_name, best_score, _ = candidates[0]
            runner_up = candidates[1][1] if len(candidates) > 1 else 0.0
            if best_score - runner_up < fuzzy_margin:
                fuzzy_by_normalized[normalized] = 0
                return 0
            fuzzy_by_normalized[normalized] = searchable[best_name]
            return fuzzy_by_normalized[normalized]

        def fuzzy_components(value: str) -> tuple[int, ...]:
            return _resolve_components(value, fuzzy_id)

        unmatched = [name for name in raw_names if mappings[name] == (0,)]
        for raw_name in tqdm(unmatched, desc="Fuzzy matching FAERS names", unit="names"):
            if raw_name in conflicts:
                continue
            prod_matches = {
                match
                for value in raw_to_prod_ai[raw_name]
                if (match := fuzzy_components(value))
            }
            if len(prod_matches) > 1:
                conflicts.add(raw_name)
                continue
            match = next(iter(prod_matches)) if prod_matches else fuzzy_components(raw_name)
            if match:
                mappings[raw_name] = match
                fuzzy_matches += 1
    unmapped = sum(dc_ids == (0,) for dc_ids in mappings.values())
    return mappings, {
        "exact_matches": exact,
        "fuzzy_matches": fuzzy_matches,
        "unmapped": unmapped,
        "exact_hit_rate": exact_rate,
        "fuzzy_enabled": fuzzy_enabled,
    }


def _resolve_components(value: str, resolver: Callable[[str], int]) -> tuple[int, ...]:
    """Resolve one name or every explicitly separated ingredient without partial matches."""
    raw_parts = re.split(r"\s*(?:/|\+)\s*", value)
    normalized_parts = [normalize_drug_name(part) for part in raw_parts]
    normalized_parts = [part for part in normalized_parts if part]
    if not normalized_parts:
        return ()
    resolved = [int(resolver(part)) for part in normalized_parts]
    if any(dc_id == 0 for dc_id in resolved):
        return ()
    return tuple(sorted(set(resolved)))


def _exact_alias_id(
    normalized: str, canonical: dict[str, set[int]], synonyms: dict[str, set[int]]
) -> int:
    """Return one unambiguous exact ID, preferring canonical names over synonyms."""
    canonical_ids = canonical.get(normalized, set())
    if canonical_ids:
        return _single_id(canonical_ids)
    return _single_id(synonyms.get(normalized, set()))


def _single_id(values: object) -> int:
    return next(iter(values)) if isinstance(values, set) and len(values) == 1 else 0


def _build_boxed_warning_flags(
    *,
    data_dir: Path,
    source: Path,
    source_sha256: str,
    memory_limit: str,
    threads: int | None,
) -> Path:
    output_dir = data_dir / "interim" / "drugcentral"
    destination = output_dir / "boxed_warning.parquet"
    manifest = output_dir / "boxed_warning_manifest.json"
    if destination.is_file() and manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("source_sha256") == source_sha256:
                _verify_boxed_warning(destination)
                return destination
        except (json.JSONDecodeError, OSError, DrugDictionaryBuildError):
            pass

    projections = {
        "active_ingredient": ("ndc_product_code", "struct_id"),
        "prd2label": ("ndc_product_code", "label_id"),
        "section": ("label_id", "code", "title"),
    }
    projected = _project_copy_tables(source, output_dir=output_dir, projections=projections)
    temporary = destination.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    connection = duckdb.connect()
    try:
        configure_duckdb(
            connection,
            data_dir=data_dir,
            stage="boxed-warning",
            memory_limit=memory_limit,
            threads=threads,
        )
        active = _sql_literal(projected["active_ingredient"].as_posix())
        labels = _sql_literal(projected["prd2label"].as_posix())
        sections = _sql_literal(projected["section"].as_posix())
        output = _sql_literal(temporary.as_posix())
        connection.execute(
            f"""
            COPY (
                SELECT DISTINCT try_cast(a.struct_id AS BIGINT) AS dc_id,
                       1::INTEGER AS has_boxed_warning
                FROM read_parquet('{active}') a
                JOIN read_parquet('{labels}') p USING (ndc_product_code)
                JOIN read_parquet('{sections}') s USING (label_id)
                WHERE try_cast(a.struct_id AS BIGINT) IS NOT NULL
                  AND (trim(s.code) = '{BOXED_WARNING_LOINC}'
                       OR upper(trim(s.title)) LIKE '%BOXED WARNING%')
                ORDER BY dc_id
            ) TO '{output}' (FORMAT PARQUET, COMPRESSION SNAPPY)
            """
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
        for path in projected.values():
            path.unlink(missing_ok=True)
    _verify_boxed_warning(temporary)
    os.replace(temporary, destination)
    payload = {
        "dataset": "DrugCentral DailyMed-derived boxed-warning flags",
        "source_path": str(source),
        "source_sha256": source_sha256,
        "boxed_warning_loinc": BOXED_WARNING_LOINC,
        "rows": pq.ParquetFile(destination).metadata.num_rows,
        "built_at_utc": datetime.now(UTC).isoformat(),
    }
    temp_manifest = manifest.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_manifest, manifest)
    return destination


def _project_copy_tables(
    source: Path, *, output_dir: Path, projections: dict[str, tuple[str, ...]]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    states: dict[str, dict[str, object]] = {}
    found: set[str] = set()
    current: str | None = None
    try:
        with gzip.open(source, mode="rt", encoding="utf-8", newline="") as stream:
            progress = tqdm(desc="Extracting DailyMed warning links", unit="lines", unit_scale=True)
            try:
                for line in stream:
                    progress.update()
                    if current is None:
                        match = _COPY_RE.match(line.rstrip("\r\n"))
                        if match is None:
                            continue
                        table = match.group("table").lower()
                        if table not in projections:
                            continue
                        source_columns = [
                            _identifier(value) for value in match.group("columns").split(",")
                        ]
                        selected = projections[table]
                        missing = set(selected) - set(source_columns)
                        if missing:
                            raise DrugDictionaryBuildError(
                                f"missing {sorted(missing)} in DrugCentral table {table}"
                            )
                        indices = [source_columns.index(column) for column in selected]
                        path = output_dir / f".{table}-{os.getpid()}.parquet.tmp"
                        path.unlink(missing_ok=True)
                        schema = pa.schema([pa.field(column, pa.string()) for column in selected])
                        states[table] = {
                            "path": path,
                            "writer": pq.ParquetWriter(path, schema, compression="snappy"),
                            "indices": indices,
                            "source_columns": source_columns,
                            "columns": selected,
                            "buffers": [[] for _ in selected],
                        }
                        found.add(table)
                        current = table
                        continue
                    if line.rstrip("\r\n") == r"\.":
                        _flush_projection(states[current])
                        writer = states[current]["writer"]
                        assert isinstance(writer, pq.ParquetWriter)
                        writer.close()
                        states[current]["writer"] = None
                        current = None
                        continue
                    _append_projection(states[current], line)
            finally:
                progress.close()
        missing_tables = set(projections) - found
        if missing_tables:
            raise DrugDictionaryBuildError(
                f"missing DrugCentral COPY tables: {sorted(missing_tables)}"
            )
        return {table: state["path"] for table, state in states.items()}
    except BaseException:
        for state in states.values():
            writer = state.get("writer")
            if isinstance(writer, pq.ParquetWriter):
                writer.close()
            path = state.get("path")
            if isinstance(path, Path):
                path.unlink(missing_ok=True)
        raise


def _append_projection(state: dict[str, object], line: str) -> None:
    values = line.rstrip("\r\n").split("\t")
    source_columns = state["source_columns"]
    indices = state["indices"]
    buffers = state["buffers"]
    assert isinstance(source_columns, list)
    assert isinstance(indices, list)
    assert isinstance(buffers, list)
    if len(values) != len(source_columns):
        raise DrugDictionaryBuildError("invalid PostgreSQL COPY row width")
    for buffer, index in zip(buffers, indices, strict=True):
        value = values[index]
        buffer.append(None if value == r"\N" else _unescape_copy(value))
    if len(buffers[0]) >= _BATCH_ROWS:
        _flush_projection(state)


def _flush_projection(state: dict[str, object]) -> None:
    buffers = state["buffers"]
    columns = state["columns"]
    writer = state["writer"]
    assert isinstance(buffers, list)
    assert isinstance(columns, tuple)
    assert isinstance(writer, pq.ParquetWriter)
    if not buffers[0]:
        return
    writer.write_batch(
        pa.record_batch([pa.array(values, type=pa.string()) for values in buffers], names=columns)
    )
    for values in buffers:
        values.clear()


def _verify_boxed_warning(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.names != ["dc_id", "has_boxed_warning"]:
        raise DrugDictionaryBuildError(f"invalid boxed-warning schema: {path}")
    _verify_snappy(path)


def _verify_dictionary(path: Path, *, expected_rows: int) -> None:
    parquet = pq.ParquetFile(path)
    expected_columns = ["faers_raw", "dc_id", "atc_code", "ror", "has_boxed_warning"]
    if parquet.schema_arrow.names != expected_columns:
        raise DrugDictionaryBuildError(f"invalid dictionary schema: {parquet.schema_arrow.names}")
    if parquet.metadata.num_rows != expected_rows:
        raise DrugDictionaryBuildError("dictionary row-count mismatch")
    table = pq.read_table(path, columns=["faers_raw", "dc_id"])
    pairs = set(zip(table["faers_raw"].to_pylist(), table["dc_id"].to_pylist(), strict=True))
    if table.num_rows != len(pairs):
        raise DrugDictionaryBuildError("duplicate faers_raw/dc_id linkage found")
    if any(value is None or value < 0 for value in table["dc_id"].to_pylist()):
        raise DrugDictionaryBuildError("invalid dc_id found")
    _verify_snappy(path)


def _verify_snappy(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    for row_group in range(parquet.metadata.num_row_groups):
        for column in range(parquet.metadata.num_columns):
            if parquet.metadata.row_group(row_group).column(column).compression != "SNAPPY":
                raise DrugDictionaryBuildError(f"non-Snappy column found in {path}")


def _validate_thresholds(trigger: float, score: float, margin: float) -> None:
    if not 0 <= trigger <= 1:
        raise ValueError("fuzzy trigger rate must be between 0 and 1")
    if not 0 <= score <= 100:
        raise ValueError("fuzzy score cutoff must be between 0 and 100")
    if not 0 <= margin <= 100:
        raise ValueError("fuzzy margin must be between 0 and 100")


def _cached_record(
    *,
    manifest: Path,
    output: Path,
    source_checksums: dict[str, str],
    fuzzy_trigger_rate: float,
    fuzzy_score_cutoff: float,
    fuzzy_margin: float,
) -> DrugDictionaryBuildRecord | None:
    if not manifest.is_file() or not output.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("build_version") != DRUG_DICTIONARY_BUILD_VERSION
            or payload.get("input_sha256") != source_checksums
            or payload.get("fuzzy_trigger_rate") != fuzzy_trigger_rate
            or payload.get("fuzzy_score_cutoff") != fuzzy_score_cutoff
            or payload.get("fuzzy_margin") != fuzzy_margin
        ):
            return None
        _verify_dictionary(output, expected_rows=int(payload["record"]["rows"]))
        record = dict(payload["record"])
        record["cached"] = True
        return DrugDictionaryBuildRecord(**record)
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        DrugDictionaryBuildError,
    ):
        return None


def _write_manifest(
    path: Path,
    *,
    record: DrugDictionaryBuildRecord,
    source_checksums: dict[str, str],
    fuzzy_trigger_rate: float,
    fuzzy_score_cutoff: float,
    fuzzy_margin: float,
) -> None:
    payload = {
        "dataset": "TekaRx FAERS drug dictionary",
        "build_version": DRUG_DICTIONARY_BUILD_VERSION,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "input_sha256": source_checksums,
        "normalization": (
            "uppercase, dosage/form/release/salt removal, explicit slash/plus splitting"
        ),
        "mapping_priority": (
            "exact prod_ai, exact drugname, exact RxNorm aliases, fuzzy prod_ai, "
            "fuzzy drugname; explicit combinations may map to multiple dc_ids"
        ),
        "ror_definition": (
            "serious-vs-nonserious reporting odds ratio over distinct report/dc_id exposures"
        ),
        "ror_zero_cell_policy": "Haldane-Anscombe 0.5 correction only when any cell is zero",
        "boxed_warning_source": ("DrugCentral DailyMed-derived SPL section table; LOINC 34066-1"),
        "fuzzy_trigger_rate": fuzzy_trigger_rate,
        "fuzzy_score_cutoff": fuzzy_score_cutoff,
        "fuzzy_margin": fuzzy_margin,
        "record": asdict(record),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")
