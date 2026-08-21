from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tekarx.transform.tabular_features import add_tabular_features


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


def test_add_tabular_features_uses_train_ror_and_atc_drug_counts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    _write(
        processed / "tekarx_cohort.parquet",
        [
            {
                "primaryid": "p1",
                "report_date": date(2024, 1, 1),
                "age": 70,
                "age_years": 70.0,
                "sex": "F",
                "weight": "70",
                "weight_unit": "KG",
                "age_missing": 0,
                "sex_unknown": 0,
                "drug_list_str": "A|B",
                "num_drugs": 2,
                "is_serious": 1,
                "split": "train",
            },
            {
                "primaryid": "p2",
                "report_date": date(2024, 1, 2),
                "age": None,
                "age_years": None,
                "sex": "UNKNOWN",
                "weight": "154",
                "weight_unit": "LBS",
                "age_missing": 1,
                "sex_unknown": 1,
                "drug_list_str": "A",
                "num_drugs": 1,
                "is_serious": 0,
                "split": "train",
            },
            {
                "primaryid": "p3",
                "report_date": date(2024, 2, 1),
                "age": 10,
                "age_years": 10.0,
                "sex": None,
                "weight": None,
                "weight_unit": None,
                "age_missing": 0,
                "sex_unknown": 1,
                "drug_list_str": "B|C",
                "num_drugs": 2,
                "is_serious": 0,
                "split": "validation",
            },
        ],
    )
    _write(
        processed / "drug_dictionary.parquet",
        [
            {
                "faers_raw": "A",
                "dc_id": 1,
                "atc_code": "N01",
                "ror": 100.0,
                "has_boxed_warning": 1,
            },
            {
                "faers_raw": "B",
                "dc_id": 2,
                "atc_code": "N02|C01",
                "ror": 0.01,
                "has_boxed_warning": 0,
            },
            {
                "faers_raw": "C",
                "dc_id": 3,
                "atc_code": "A10",
                "ror": 50.0,
                "has_boxed_warning": 0,
            },
        ],
    )
    _write(
        processed / "edges" / "report_drug.parquet",
        [
            {
                "primaryid": "p1",
                "drugname": "A",
                "route": "ORAL",
                "dose_amt": "10",
                "dose_unit": "MG",
                "dose_form": "TABLET",
                "dose_freq": "DAILY",
            },
            {
                "primaryid": "p1",
                "drugname": "B",
                "route": "INTRAVENOUS",
                "dose_amt": None,
                "dose_unit": None,
                "dose_form": "INJECTION",
                "dose_freq": None,
            },
            {
                "primaryid": "p2",
                "drugname": "A",
                "route": None,
                "dose_amt": None,
                "dose_unit": None,
                "dose_form": None,
                "dose_freq": None,
            },
            {
                "primaryid": "p3",
                "drugname": "B",
                "route": "ORAL",
                "dose_amt": "5",
                "dose_unit": "MG",
                "dose_form": "CAPSULE",
                "dose_freq": "BID",
            },
            {
                "primaryid": "p3",
                "drugname": "C",
                "route": "TOPICAL",
                "dose_amt": None,
                "dose_unit": None,
                "dose_form": "SOLUTION",
                "dose_freq": None,
            },
        ],
    )

    record = add_tabular_features(data_dir=data_dir, rebuild_graph=False, memory_limit="512MB")
    rows = {
        row["primaryid"]: row
        for row in pq.read_table(record.output_path).to_pylist()
    }

    assert "strict-prior-date" in record.ror_scope
    assert rows["p1"]["max_ror"] == 1.0
    assert rows["p1"]["high_ror_count"] == 0
    assert rows["p1"]["has_boxed_warning"] == 1
    assert rows["p1"]["num_high_risk_atc"] == 2
    assert rows["p1"]["atc_diversity"] == 2
    assert rows["p1"]["therapeutic_duplicates"] == 1
    assert rows["p1"]["atc_l1_count_n"] == 2
    assert rows["p1"]["atc_l1_count_c"] == 1
    assert rows["p1"]["exposure_route_diversity"] == 2
    assert rows["p1"]["exposure_has_oral_route"] == 1
    assert rows["p1"]["exposure_has_parenteral_route"] == 1
    assert rows["p1"]["exposure_dose_documented_fraction"] == 0.5
    assert rows["p1"]["age_group_65_plus"] == 1
    assert rows["p1"]["num_drugs_squared"] == 4
    assert rows["p1"]["polypharmacy_age"] == 140.0
    assert rows["p2"]["max_ror"] == 1.0
    assert rows["p2"]["age_missing"] == 1
    assert rows["p2"]["age_imputed_years"] == 70.0
    assert rows["p2"]["age_group_65_plus"] == 1
    assert rows["p2"]["sex_unknown"] == 1
    assert rows["p2"]["weight_missing"] == 0
    assert rows["p3"]["max_ror"] == 9.0
    assert rows["p3"]["atc_diversity"] == 3
    assert rows["p3"]["age_group_0_17"] == 1
    assert rows["p3"]["weight_missing"] == 1
