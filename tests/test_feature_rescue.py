from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tekarx.transform import feature_rescue


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


def test_feature_rescue_freezes_train_pairs_and_auxiliary_outcomes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    cohort_rows = [
        {
            "primaryid": "p1",
            "report_date": date(2023, 1, 1),
            "drug_list_str": "A|B",
            "outcome_codes": "DE|HO",
            "is_serious": 1,
            "split": "train",
        },
        {
            "primaryid": "p2",
            "report_date": date(2023, 2, 1),
            "drug_list_str": "A",
            "outcome_codes": "OT",
            "is_serious": 0,
            "split": "train",
        },
        {
            "primaryid": "p3",
            "report_date": date(2023, 3, 1),
            "drug_list_str": "C",
            "outcome_codes": "OT",
            "is_serious": 0,
            "split": "train",
        },
        {
            "primaryid": "p4",
            "report_date": date(2023, 4, 1),
            "drug_list_str": "C",
            "outcome_codes": "LT",
            "is_serious": 1,
            "split": "train",
        },
        {
            "primaryid": "p5",
            "report_date": date(2024, 1, 1),
            "drug_list_str": "A|B",
            "outcome_codes": "HO",
            "is_serious": 1,
            "split": "validation",
        },
    ]
    _write(processed / "tekarx_cohort.parquet", cohort_rows)
    _write(
        processed / "drug_dictionary.parquet",
        [
            {"faers_raw": "A", "dc_id": 1},
            {"faers_raw": "B", "dc_id": 2},
            {"faers_raw": "C", "dc_id": 3},
        ],
    )
    _write(
        data_dir / "interim" / "faers" / "indi" / "2023Q1.parquet",
        [
            {"primaryid": "p1", "indi_pt": "Lung cancer"},
            {"primaryid": "p2", "indi_pt": "Bacterial infection"},
            {"primaryid": "p3", "indi_pt": "Heart failure"},
            {"primaryid": "p5", "indi_pt": "Lung cancer"},
            {"primaryid": "p5", "indi_pt": "Novel heldout disease"},
        ],
    )
    record = feature_rescue.build_feature_rescue(
        data_dir=data_dir,
        top_pairs=1,
        minimum_pair_reports=1,
        memory_limit="512MB",
        rebuild_graph=False,
    )
    rows = {
        row["primaryid"]: row for row in pq.read_table(record.output_path).to_pylist()
    }
    pair = pq.read_table(record.pair_lookup_path).to_pylist()[0]

    assert pair["dc_id_1"] == 1 and pair["dc_id_2"] == 2
    assert record.graph_path is None
    assert record.validation_auc is None
    assert pair["prr"] == 3.0
    assert rows["p5"]["num_high_risk_pairs"] == 1
    assert rows["p1"]["num_high_risk_pairs"] == 0
    assert rows["p1"]["scored_pair_count"] == 0
    assert rows["p5"]["scored_pair_count"] == 1
    assert rows["p5"]["max_pair_log_ror"] > 1.0
    assert rows["p5"]["has_malignancy"] == 1
    assert sum(rows["p5"][f"indication_hash_{index:02d}"] for index in range(32)) == 1
    assert "reporter_ph" not in rows["p5"]
    assert rows["p1"]["is_death"] == 1
    assert rows["p1"]["is_hospitalization"] == 1
    assert "reporter_unknown" not in rows["p4"]
