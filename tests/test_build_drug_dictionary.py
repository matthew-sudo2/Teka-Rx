import gzip
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tekarx.transform.drug_dictionary import (
    _map_names,
    build_drug_dictionary,
    normalize_drug_name,
)
from tekarx.transform.rxnorm_lookup import _api_lookup_rows, build_rxnorm_lookup


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="snappy")


def _write_dictionary_fixture(data_dir: Path) -> None:
    processed = data_dir / "processed"
    _write_table(
        processed / "tekarx_cohort.parquet",
        [
            {"primaryid": "1", "is_serious": 1},
            {"primaryid": "2", "is_serious": 0},
            {"primaryid": "3", "is_serious": 0},
            {"primaryid": "4", "is_serious": 1},
            {"primaryid": "5", "is_serious": 1},
        ],
    )
    _write_table(
        processed / "edges" / "report_drug.parquet",
        [
            {"primaryid": "1", "drugname": "ASPIRIN 500MG", "prod_ai": "NOT LISTED"},
            {"primaryid": "2", "drugname": "ASPIRIN 500MG", "prod_ai": None},
            {"primaryid": "3", "drugname": "BRAND X", "prod_ai": "METFORMIN"},
            {"primaryid": "4", "drugname": "ASPIRIN BRAND", "prod_ai": "ASPIRIN"},
            {"primaryid": "5", "drugname": "UNKNOWN PRODUCT", "prod_ai": None},
        ],
    )
    drugcentral = data_dir / "interim" / "drugcentral"
    _write_table(
        drugcentral / "structures.parquet",
        [{"id": "1", "name": "Aspirin"}, {"id": "2", "name": "Metformin"}],
    )
    _write_table(
        drugcentral / "synonyms.parquet",
        [
            {"id": "1", "name": "Acetylsalicylic acid"},
            {"id": "2", "name": "Metformin hydrochloride"},
        ],
    )
    _write_table(
        drugcentral / "struct2atc.parquet",
        [
            {"struct_id": "1", "atc_code": "B01AC06"},
            {"struct_id": "1", "atc_code": "N02BA01"},
            {"struct_id": "2", "atc_code": "A10BA02"},
        ],
    )
    raw = data_dir / "raw" / "drugcentral"
    raw.mkdir(parents=True)
    sql = "\n".join(
        [
            "COPY public.active_ingredient (id, ndc_product_code, struct_id) FROM stdin;",
            "1\t0001-0001\t1",
            "2\t0002-0002\t2",
            r"\.",
            "COPY public.prd2label (ndc_product_code, label_id, id) FROM stdin;",
            "0001-0001\tlabel-1\t1",
            "0002-0002\tlabel-2\t2",
            r"\.",
            "COPY public.section (id, text, label_id, code, title) FROM stdin;",
            "1\tWarning text\tlabel-1\t34066-1\tBOXED WARNING SECTION",
            "2\tOther text\tlabel-2\t34084-4\tADVERSE REACTIONS",
            r"\.",
            "COPY public.identifier (id, identifier, id_type, struct_id, parent_match) FROM stdin;",
            "1\t1191\t36\t1\tf",
            r"\.",
            "COPY public.id_type (id, type, description, url) FROM stdin;",
            "36\tRXNORM\tRxNorm Vocabulary\thttps://example.test/rxnorm",
            r"\.",
            "",
        ]
    )
    with gzip.open(raw / "fixture.sql.gz", "wt", encoding="utf-8", newline="") as stream:
        stream.write(sql)


def test_normalize_drug_name_removes_dose_form_and_release_suffixes() -> None:
    assert normalize_drug_name("Metformin ER 500MG") == "METFORMIN"
    assert normalize_drug_name("Metformin 500MG XR") == "METFORMIN"
    assert normalize_drug_name("Aspirin (CAPSULE)") == "ASPIRIN"
    assert normalize_drug_name("Metformin Hydrochloride 500 MG Oral Tablet") == "METFORMIN"


def test_mapping_priority_uses_prod_ai_before_raw_and_fuzzy_fallbacks() -> None:
    aliases = {
        "canonical": {
                "ASPIRIN": {1},
                "METFORMIN": {2},
                "ACETYLSALICYLIC ACID": {1},
        },
        "synonyms": {},
    }
    mappings, stats = _map_names(
        {
            "ASPIRIN": ["METFORMIN"],
            "RAW BRAND": ["ACETYLSSALICYLIC ACID"],
            "ACETYLSSALICYLIC ACID": [],
            "AMBIGUOUS COMBINATION": ["ASPIRIN", "METFORMIN"],
            "EXPLICIT COMBINATION": ["ASPIRIN + METFORMIN"],
            "RXNORM BRAND": [],
            "RXNORM COMBO": [],
        },
        aliases=aliases,
        rxnorm_aliases={"RXNORM BRAND": {2}, "RXNORM COMBO": {1, 2}},
        fuzzy_trigger_rate=1.0,
        fuzzy_score_cutoff=97.0,
        fuzzy_margin=3.0,
    )

    assert mappings["ASPIRIN"] == (2,)
    assert mappings["RAW BRAND"] == (1,)
    assert mappings["ACETYLSSALICYLIC ACID"] == (1,)
    assert mappings["AMBIGUOUS COMBINATION"] == (0,)
    assert mappings["EXPLICIT COMBINATION"] == (1, 2)
    assert mappings["RXNORM BRAND"] == (2,)
    assert mappings["RXNORM COMBO"] == (1, 2)
    assert stats["exact_matches"] == 4
    assert stats["fuzzy_matches"] == 2


def test_rxnorm_api_candidate_explodes_combination_and_strips_salts() -> None:
    rows = _api_lookup_rows(
        {
            "JANUMET": [
                {
                    "rxcui": "123",
                    "name": (
                        "sitagliptin 50 MG / metformin hydrochloride 500 MG "
                        "Oral Tablet [Janumet]"
                    ),
                }
            ]
        },
        aliases={
            "canonical": {"SITAGLIPTIN": {3}, "METFORMIN": {2}},
            "synonyms": {},
        },
    )

    assert {row["struct_id"] for row in rows} == {2, 3}
    assert {row["query_name"] for row in rows} == {"JANUMET"}


def test_build_rxnorm_lookup_uses_generic_drugcentral_identifiers(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_dictionary_fixture(data_dir)
    _write_table(
        data_dir / "interim" / "dailymed" / "rxnorm.parquet",
        [{"rxcui": "1191", "rxstring": "Aspirin Brand"}],
    )

    record = build_rxnorm_lookup(data_dir=data_dir)

    assert record.rxnorm_identifier_type_found
    assert not record.rxnorm_table_found
    assert record.local_rows == 1
    lookup = pq.read_table(record.output_path).to_pylist()
    assert lookup == [
        {
            "rxnorm_name": "Aspirin Brand",
            "struct_id": 1,
            "rxcui": "1191",
            "source": "drugcentral_identifier",
            "query_name": None,
        }
    ]


def test_build_drug_dictionary_exact_mapping_ror_and_boxed_warning(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_dictionary_fixture(data_dir)

    first = build_drug_dictionary(data_dir=data_dir, memory_limit="512MB")
    second = build_drug_dictionary(data_dir=data_dir, memory_limit="512MB")

    assert not first.cached
    assert second.cached
    assert first.exact_matches == 3
    assert first.fuzzy_matches == 0
    assert first.unmapped == 1
    assert first.exact_hit_rate == 3 / 4
    assert not first.fuzzy_enabled
    output = pq.read_table(data_dir / "processed" / "drug_dictionary.parquet").to_pandas()
    assert list(output.columns) == [
        "faers_raw",
        "dc_id",
        "atc_code",
        "ror",
        "has_boxed_warning",
    ]
    assert output["faers_raw"].is_unique
    indexed = output.set_index("faers_raw")
    aspirin = indexed.loc["ASPIRIN 500MG"]
    assert aspirin["dc_id"] == 1
    assert aspirin["atc_code"] == "B01AC06|N02BA01"
    assert aspirin["ror"] == 2.0
    assert aspirin["has_boxed_warning"] == 1
    aspirin_alias = indexed.loc["ASPIRIN BRAND"]
    assert aspirin_alias["dc_id"] == 1
    assert aspirin_alias["ror"] == aspirin["ror"]
    metformin = indexed.loc["BRAND X"]
    assert metformin["dc_id"] == 2
    assert metformin["has_boxed_warning"] == 0
    assert indexed.loc["UNKNOWN PRODUCT", "dc_id"] == 0
