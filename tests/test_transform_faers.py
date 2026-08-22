import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tekarx.extract.common import sha256_file
from tekarx.transform.faers import build_faers


def _write_faers_fixture(data_dir: Path, quarter: str) -> None:
    extracted = data_dir / "raw" / "faers" / quarter / "extracted"
    ascii_dir = extracted / "ASCII"
    deleted_dir = extracted / "Deleted"
    ascii_dir.mkdir(parents=True)
    deleted_dir.mkdir(parents=True)
    (extracted / ".complete").write_text("verified extraction\n", encoding="utf-8")
    fixtures = {
        ascii_dir / "DEMO24Q1.txt": (
            "primaryid$caseid$caseversion$age$age_cod$sex$wt$wt_cod\n"
            "1001$100$1$42$YR$F$60$KG\n"
            "1002$101$2$$YR$M$$\n"
        ),
        ascii_dir / "DRUG24Q1.txt": (
            "primaryid$caseid$drug_seq$drugname$route$dose_amt$dose_unit\n"
            "1001$100$1$ASPIRIN$Oral$10$MG\n"
        ),
        ascii_dir / "INDI24Q1.txt": (
            "primaryid$caseid$indi_drug_seq$indi_pt\n1001$100$1$Pain\n"
        ),
        ascii_dir / "REAC24Q1.txt": "primaryid$caseid$pt\n1001$100$Headache\n",
        ascii_dir / "OUTC24Q1.txt": "primaryid$caseid$outc_cod\n1001$100$HO\n",
        deleted_dir / "DELETE24Q1.txt": " \n999\n",
    }
    for path, content in fixtures.items():
        path.write_text(content, encoding="utf-8")


def test_build_faers_streams_core_tables_to_snappy_parquet(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_faers_fixture(data_dir, "2024Q1")

    first = build_faers(data_dir=data_dir, quarters=["2024Q1"])
    second = build_faers(data_dir=data_dir, quarters=["2024Q1"])

    assert len(first) == 6
    assert all(not record.cached for record in first)
    assert all(record.cached for record in second)
    demo_path = data_dir / "interim" / "faers" / "demo" / "2024Q1.parquet"
    demo = pq.read_table(demo_path)
    assert demo.column_names == [
        "primaryid",
        "caseid",
        "caseversion",
        "age",
        "age_cod",
        "sex",
        "wt",
        "wt_cod",
        "quarter",
    ]
    assert demo.schema.field("primaryid").type == pa.string()
    assert demo["primaryid"].to_pylist() == ["1001", "1002"]
    assert demo["age"].to_pylist() == ["42", None]
    assert demo["quarter"].to_pylist() == ["2024Q1", "2024Q1"]
    parquet = pq.ParquetFile(demo_path)
    assert parquet.metadata.row_group(0).column(0).compression == "SNAPPY"
    assert parquet.schema_arrow.metadata[b"tekarx.table"] == b"demo"
    deleted = pq.read_table(data_dir / "interim/faers/delete/2024Q1.parquet")
    assert deleted.column_names == ["caseid", "quarter"]
    assert deleted["caseid"].to_pylist() == ["999"]
    assert (data_dir / "interim" / "faers" / "manifest.json").is_file()


def test_build_faers_combines_historical_deletion_file_names(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    extracted = data_dir / "raw" / "faers" / "2019Q1" / "extracted"
    deleted_dir = extracted / "Deleted"
    deleted_dir.mkdir(parents=True)
    (extracted / ".complete").write_text("verified extraction\n", encoding="utf-8")
    (deleted_dir / "ADR19Q1DeletedCases.txt").write_text(
        "caseid\n10417202\n999$\n", encoding="utf-8"
    )
    (deleted_dir / "AllDeletedCases.txt").write_text(
        "\n820242\n10417202\n", encoding="utf-8"
    )
    (deleted_dir / "README.txt").write_text(
        "Deletion file documentation for 2019.\n", encoding="utf-8"
    )

    first = build_faers(data_dir=data_dir, quarters=["2019Q1"], tables=("delete",))
    second = build_faers(data_dir=data_dir, quarters=["2019Q1"], tables=("delete",))

    assert len(first) == 1
    assert first[0].cached is False
    assert second[0].cached is True
    deleted_path = data_dir / "interim" / "faers" / "delete" / "2019Q1.parquet"
    deleted = pq.read_table(deleted_path)
    assert deleted["caseid"].to_pylist() == ["10417202", "999", "820242", "10417202"]
    metadata = deleted.schema.metadata or {}
    sources = json.loads(metadata[b"tekarx.source_path"])
    assert [Path(item["path"]).name for item in sources] == [
        "ADR19Q1DeletedCases.txt",
        "AllDeletedCases.txt",
    ]


def test_build_faers_records_an_empty_delete_table_when_archive_has_no_delete_file(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    extracted = data_dir / "raw" / "faers" / "2019Q1" / "extracted"
    extracted.mkdir(parents=True)
    (extracted / ".complete").write_text("verified extraction\n", encoding="utf-8")
    archive = data_dir / "raw" / "faers" / "2019Q1" / "source.zip"
    archive.write_bytes(b"official archive fixture")

    first = build_faers(data_dir=data_dir, quarters=["2019Q1"], tables=("delete",))
    second = build_faers(data_dir=data_dir, quarters=["2019Q1"], tables=("delete",))

    assert first[0].rows == 0
    source_entries = json.loads(first[0].source_path)
    assert source_entries == [
        {
            "path": "raw/faers/2019Q1/source.zip",
            "role": "archive_without_delete_file",
            "sha256": sha256_file(archive),
        }
    ]
    assert first[0].cached is False
    assert second[0].cached is True
    deleted_path = data_dir / "interim" / "faers" / "delete" / "2019Q1.parquet"
    deleted = pq.read_table(deleted_path)
    assert deleted.column_names == ["caseid", "quarter"]
    assert deleted.num_rows == 0
    metadata = deleted.schema.metadata or {}
    assert json.loads(metadata[b"tekarx.source_path"]) == source_entries
