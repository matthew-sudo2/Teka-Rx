import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tekarx.transform.cohort import build_cohort


def _write_table(path: Path, rows: list[dict[str, str | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    table = pa.table(
        {column: pa.array([row[column] for row in rows], type=pa.string()) for column in columns}
    )
    pq.write_table(table, path, compression="snappy")


def _write_cohort_fixture(data_dir: Path) -> None:
    root = data_dir / "interim" / "faers"
    demo_rows = [
        _demo("1001", "100", "1", "12", "MON", "20230101", "2023Q1"),
        _demo("1002", "100", "2", "2", "DEC", "20231201", "2023Q4"),
        _demo("2001", "200", "1", "72", "YR", "20240201", "2024Q1"),
        _demo("3001", "300", "1", "600", "MON", "20240501", "2024Q2"),
        _demo("4001", "400", "1", "520", "WK", "20230501", "2023Q2"),
        _demo("5001", "500", "1", "3652.5", "DY", "20230901", "2023Q3"),
        _demo("6001", "600", "1", "40", "YR", "20230701", "2023Q3"),
        _demo("7001", "700", "1", "13", "DEC", "20230801", "2023Q3"),
        _demo("8001", "800", "1", None, None, "20230301", "2023Q1", sex=None),
        _demo("9001", "900", "1", "30", "YR", "20231001", "2023Q4"),
    ]
    _write_table(root / "demo" / "fixture.parquet", demo_rows)

    drug_rows = []
    for primaryid, caseid, quarter in [
        ("1001", "100", "2023Q1"),
        ("1002", "100", "2023Q4"),
        ("2001", "200", "2024Q1"),
        ("3001", "300", "2024Q2"),
        ("4001", "400", "2023Q2"),
        ("5001", "500", "2023Q3"),
        ("6001", "600", "2023Q3"),
        ("7001", "700", "2023Q3"),
        ("8001", "800", "2023Q1"),
    ]:
        drug_rows.append(_drug(primaryid, caseid, "1", f"Drug {primaryid}", quarter))
    drug_rows.append(_drug("1002", "100", "2", "Second Drug", "2023Q4"))
    drug_rows.append(_drug("1002", "100", "2", "Second Drug", "2023Q4"))
    _write_table(root / "drug" / "fixture.parquet", drug_rows)

    report_quarters = {row["primaryid"]: row["quarter"] for row in demo_rows}
    reaction_rows = [
        {
            "primaryid": primaryid,
            "caseid": caseid,
            "pt": "Headache",
            "drug_rec_act": None,
            "quarter": report_quarters[primaryid],
        }
        for primaryid, caseid in [
            ("1001", "100"),
            ("1002", "100"),
            ("2001", "200"),
            ("3001", "300"),
            ("4001", "400"),
            ("5001", "500"),
        ]
    ]
    reaction_rows.append(reaction_rows[1].copy())
    _write_table(root / "reac" / "fixture.parquet", reaction_rows)

    outcome_rows = []
    codes = {
        "1001": ["OT"],
        "1002": ["OT", "HO"],
        "2001": ["DE", "OT"],
        "3001": ["OT"],
        "4001": ["CA"],
        "5001": ["LT"],
        "6001": ["HO"],
        "7001": ["HO"],
        "9001": ["HO"],
    }
    caseids = {primaryid: primaryid[:-1] for primaryid in codes}
    for primaryid, values in codes.items():
        for code in values:
            outcome_rows.append(
                {
                    "primaryid": primaryid,
                    "caseid": caseids[primaryid],
                    "outc_cod": code,
                    "quarter": report_quarters[primaryid],
                }
            )
    outcome_rows.append(outcome_rows[2].copy())
    _write_table(root / "outc" / "fixture.parquet", outcome_rows)
    _write_table(
        root / "delete" / "fixture.parquet",
        [{"caseid": "600", "quarter": "2023Q3"}],
    )


def _demo(
    primaryid: str,
    caseid: str,
    version: str,
    age: str | None,
    unit: str | None,
    report_date: str,
    quarter: str,
    *,
    sex: str | None = "F",
) -> dict[str, str | None]:
    return {
        "primaryid": primaryid,
        "caseid": caseid,
        "caseversion": version,
        "event_dt": None,
        "fda_dt": report_date,
        "rept_dt": report_date,
        "age": age,
        "age_cod": unit,
        "sex": sex,
        "wt": "60",
        "wt_cod": "KG",
        "quarter": quarter,
    }


def _drug(
    primaryid: str, caseid: str, sequence: str, name: str, quarter: str
) -> dict[str, str | None]:
    return {
        "primaryid": primaryid,
        "caseid": caseid,
        "drug_seq": sequence,
        "role_cod": "PS",
        "drugname": name,
        "prod_ai": name,
        "route": "Oral",
        "dose_amt": "1",
        "dose_unit": "MG",
        "dose_form": "Tablet",
        "dose_freq": "QD",
        "quarter": quarter,
    }


def test_build_cohort_keeps_latest_case_and_preserves_split_edges(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_cohort_fixture(data_dir)

    first = build_cohort(data_dir=data_dir, memory_limit="512MB")
    split_plan_path = data_dir / "processed" / "splits" / "faers_gnn-small.json"
    assert split_plan_path.is_file()
    split_plan_path.unlink()
    second = build_cohort(data_dir=data_dir, memory_limit="512MB")

    assert not first.cached
    assert second.cached
    split_plan = json.loads(split_plan_path.read_text(encoding="utf-8"))
    assert split_plan["preset"] == "gnn-small"
    assert split_plan["group_key"] == "CASEID"
    assert split_plan["splits"]["validation"] == ["2024Q1"]
    assert first.source_demo_rows == 10
    assert first.source_cases == 9
    assert first.deleted_cases == 1
    assert first.nondeleted_cases == 8
    assert first.latest_cases == 8
    assert first.valid_age_cases == 7
    assert first.cases_with_drugs == 6
    assert first.cases_with_outcomes == 6
    cohort = pq.read_table(data_dir / "processed" / "tekarx_cohort.parquet").to_pandas()
    assert set(cohort["primaryid"]) == {"1002", "2001", "3001", "4001", "5001", "8001"}
    assert cohort["primaryid"].is_unique
    assert cohort["caseid"].is_unique
    latest = cohort.set_index("primaryid").loc["1002"]
    assert latest["caseversion"] == 2
    assert latest["age"] == 20
    assert latest["num_drugs"] == 2
    assert latest["outcome_codes"] == "HO|OT"
    assert latest["is_serious"] == 1
    assert cohort.set_index("primaryid").loc["3001", "age"] == 50
    indexed = cohort.set_index("primaryid")
    assert indexed.loc["3001", "is_serious"] == 1  # OT is an official serious outcome.
    assert indexed.loc["4001", "is_serious"] == 1  # CA is an official serious outcome.
    missing = indexed.loc["8001"]
    assert pd.isna(missing["age"])
    assert missing["age_missing"] == 1
    assert missing["age_group"] == "UNKNOWN"
    assert missing["sex"] == "UNKNOWN"
    assert missing["sex_unknown"] == 1
    assert pd.isna(missing["outcome_codes"])
    assert missing["is_serious"] == 0
    assert indexed.loc["2001", "split"] == "validation"
    assert indexed.loc["3001", "split"] == "test"
    assert set(indexed.loc[["1002", "4001", "5001", "8001"], "split"]) == {"train"}

    drug_edges = pq.read_table(data_dir / "processed" / "edges" / "report_drug.parquet")
    assert first.drug_edges == 7
    assert first.reaction_edges == 5
    assert first.outcome_edges == 7
    assert drug_edges.num_rows == 7
    assert "1001" not in drug_edges["primaryid"].to_pylist()
    assert set(drug_edges["primaryid"].to_pylist()) == set(cohort["primaryid"])
    case_splits = pq.read_table(data_dir / "processed" / "case_splits.parquet").to_pandas()
    assert case_splits["caseid"].is_unique
    assert "600" not in set(case_splits["caseid"])
    assert case_splits.set_index("caseid").loc["100", "quarter"] == "2023Q4"
    expected_split = {
        "2023Q1": "train",
        "2023Q2": "train",
        "2023Q3": "train",
        "2023Q4": "train",
        "2024Q1": "validation",
        "2024Q2": "test",
    }
    assert case_splits.apply(
        lambda row: row["split"] == expected_split[row["quarter"]], axis=1
    ).all()
    manifest = json.loads(
        (data_dir / "processed" / "cohort_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["split_preset"] == "gnn-small"
    assert manifest["quarter_coverage"]["missing"] == {
        "train": [],
        "validation": [],
        "test": [],
    }
    assert set(manifest["serious_outcome_codes"]) == {"DE", "LT", "HO", "DS", "RI", "CA", "OT"}


def test_incomplete_full_preset_warns_and_records_actual_coverage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    _write_cohort_fixture(data_dir)

    record = build_cohort(data_dir=data_dir, split_preset="gnn-full", memory_limit="512MB")

    captured = capsys.readouterr()
    assert record.train_rows > 0
    assert record.validation_rows > 0
    assert record.test_rows > 0
    assert "missing preset quarter(s)" in captured.err
    manifest = json.loads(
        (data_dir / "processed" / "cohort_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["quarter_coverage"]["present"]["train"] == [
        "2023Q1",
        "2023Q2",
        "2023Q3",
        "2023Q4",
    ]
    assert manifest["quarter_coverage"]["missing"]["train"] == [
        f"{year}Q{quarter}" for year in range(2019, 2023) for quarter in range(1, 5)
    ]
