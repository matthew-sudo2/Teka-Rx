import gzip
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tekarx.transform.dailymed import build_dailymed
from tekarx.transform.drugcentral import build_drugcentral


def _write_drugcentral_fixture(data_dir: Path) -> None:
    directory = data_dir / "raw" / "drugcentral"
    directory.mkdir(parents=True)
    blocks = {
        "structures": ("id, name", [r"1\tAspirin", r"2\tMetformin"]),
        "synonyms": ("id, name, preferred_name, parent_id", [r"10\tASA\tf\t1"]),
        "atc": ("id, code, chemical_substance, l2_name", [r"20\tB01AC06\taspirin\tAgents"]),
        "struct2atc": ("struct_id, atc_code", [r"1\tB01AC06"]),
        "drug_class": ("id, name, source", [r"30\tPlatelet inhibitor\tFDA"]),
    }
    lines = ["-- PostgreSQL database dump"]
    for table, (columns, values) in blocks.items():
        lines.append(f"COPY public.{table} ({columns}) FROM stdin;")
        lines.extend(value.replace(r"\t", "\t") for value in values)
        lines.append(r"\.")
    with gzip.open(directory / "fixture.sql.gz", "wt", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(lines) + "\n")


def _write_dailymed_fixtures(data_dir: Path) -> None:
    directory = data_dir / "raw" / "dailymed"
    directory.mkdir(parents=True)
    fixtures = {
        "rxnorm_mappings.zip": (
            "rxnorm_mappings.txt",
            "SETID|SPL_VERSION|RXCUI|RXSTRING|RXTTY\n"
            "set-1|2|1191|Aspirin 81 MG Oral Tablet|SCD\n",
        ),
        "pharmacologic_class_mappings.zip": (
            "pharmacologic_class_mappings.txt",
            "SPL_SETID|SPL_VERSION|PHARMA_SETID|PHARMA_VERSION\nset-1|2|class-1|1\n",
        ),
        "dm_spl_zip_files_meta_data.zip": (
            "dm_spl_zip_files_meta_data.txt",
            "SETID|ZIP_FILE_NAME|UPLOAD_DATE|SPL_VERSION|TITLE\n"
            "set-1|label.zip|01/01/2024|2|ASPIRIN [EXAMPLE]\n",
        ),
    }
    for archive_name, (member_name, content) in fixtures.items():
        with zipfile.ZipFile(directory / archive_name, "w") as archive:
            archive.writestr("README.txt", "documentation only\n")
            archive.writestr(member_name, content)


def test_build_drugcentral_streams_copy_tables_to_snappy_parquet(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_drugcentral_fixture(data_dir)

    first = build_drugcentral(data_dir=data_dir)
    second = build_drugcentral(data_dir=data_dir)

    assert len(first) == 5
    assert all(not record.cached for record in first)
    assert all(record.cached for record in second)
    structures_path = data_dir / "interim" / "drugcentral" / "structures.parquet"
    structures = pq.read_table(structures_path)
    assert structures.column_names == ["id", "name"]
    assert structures.schema.field("id").type == pa.string()
    assert structures.to_pylist() == [
        {"id": "1", "name": "Aspirin"},
        {"id": "2", "name": "Metformin"},
    ]
    parquet = pq.ParquetFile(structures_path)
    assert parquet.metadata.row_group(0).column(0).compression == "SNAPPY"
    assert parquet.schema_arrow.metadata[b"tekarx.dataset"] == b"drugcentral"
    assert (data_dir / "interim" / "drugcentral" / "manifest.json").is_file()


def test_build_dailymed_streams_all_mapping_archives_to_snappy_parquet(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_dailymed_fixtures(data_dir)

    first = build_dailymed(data_dir=data_dir)
    second = build_dailymed(data_dir=data_dir)

    assert len(first) == 3
    assert all(not record.cached for record in first)
    assert all(record.cached for record in second)
    rxnorm_path = data_dir / "interim" / "dailymed" / "rxnorm.parquet"
    rxnorm = pq.read_table(rxnorm_path)
    assert rxnorm.column_names == ["spl_setid", "spl_version", "rxcui", "rxstring", "rxtty"]
    assert rxnorm.schema.field("rxcui").type == pa.string()
    assert rxnorm["rxcui"].to_pylist() == ["1191"]
    parquet = pq.ParquetFile(rxnorm_path)
    assert parquet.metadata.row_group(0).column(0).compression == "SNAPPY"
    assert parquet.schema_arrow.metadata[b"tekarx.dataset"] == b"dailymed"
    assert (data_dir / "interim" / "dailymed" / "manifest.json").is_file()
