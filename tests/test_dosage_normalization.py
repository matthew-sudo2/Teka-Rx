from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from tekarx.transform.dosage import canonicalize_dose, frequency_per_day
from tekarx.transform.tabular_features import add_tabular_features


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


@pytest.mark.parametrize(
    ("amount", "unit", "dimension", "expected"),
    [
        (1_000, "UG", "mass_mg", 1.0),
        (1_000, "MCG", "mass_mg", 1.0),
        (1_000, "µG", "mass_mg", 1.0),
        (1_000, "μG", "mass_mg", 1.0),
        (1, "M.G.", "mass_mg", 1.0),
        (1, "G", "mass_mg", 1_000.0),
        (1, "MG/KG", "mass_mg_per_kg", 1.0),
        (1_000, "UG/M2", "mass_mg_per_m2", 1.0),
        (1, "L", "volume_ml", 1_000.0),
        (1, "KIU", "activity_iu", 1_000.0),
        (1, "IU/KG", "activity_iu_per_kg", 1.0),
        (1, "MEQ", "equivalent_meq", 1.0),
        (1_000, "UMOL", "substance_mmol", 1.0),
        (1, "GBQ", "radioactivity_mbq", 1_000.0),
    ],
)
def test_canonicalize_dose_preserves_dimensions(
    amount: float, unit: str, dimension: str, expected: float
) -> None:
    result = canonicalize_dose(amount, unit)

    assert result is not None
    assert result.dimension == dimension
    assert result.amount == pytest.approx(expected)


@pytest.mark.parametrize("amount", [0, -1, float("nan"), float("inf")])
def test_canonicalize_dose_rejects_nonpositive_or_nonfinite_values(amount: float) -> None:
    assert canonicalize_dose(amount, "MG") is None


@pytest.mark.parametrize("unit", ["%", "PCT", "GTT", "TABLET", "SPRAY", "MG/ML", ""])
def test_canonicalize_dose_refuses_unsafe_or_presentation_conversions(unit: str) -> None:
    assert canonicalize_dose(10, unit) is None


def test_canonicalize_dose_never_conflates_volume_activity_and_mass() -> None:
    mass = canonicalize_dose(1, "MG")
    volume = canonicalize_dose(1, "ML")
    activity = canonicalize_dose(1, "IU")

    assert mass is not None and volume is not None and activity is not None
    assert len({mass.dimension, volume.dimension, activity.dimension}) == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("QD", 1.0),
        ("DAILY", 1.0),
        ("BID", 2.0),
        ("TID", 3.0),
        ("QID", 4.0),
        ("Q12H", 2.0),
        ("Q8H", 3.0),
        ("QOD", 0.5),
        ("QW", 1 / 7),
        ("TIW", 3 / 7),
        ("Q3W", 1 / 21),
        ("QM", 1 / 30.4375),
        ("1/YR", 1 / 365.25),
    ],
)
def test_frequency_per_day_only_converts_unambiguous_schedules(
    value: str, expected: float
) -> None:
    converted = frequency_per_day(value)

    assert converted is not None
    assert math.isclose(converted, expected, rel_tol=1e-12)


@pytest.mark.parametrize(
    "value",
    [None, "", "PRN", "BIW", "ONCE", "1-2 TIMES DAILY", "EVERY 4-6 HOURS", "UNKNOWN"],
)
def test_frequency_per_day_rejects_ambiguous_schedules(value: str | None) -> None:
    assert frequency_per_day(value) is None


def test_dosage_lookup_is_train_frozen_and_combination_dose_is_not_duplicated(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    cohort_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    start = date(2024, 1, 1)
    for index, amount in enumerate((1, 2, 3, 4, 5), start=1):
        primaryid = f"train-{index}"
        cohort_rows.append(
            _cohort_row(
                primaryid,
                start + timedelta(days=index),
                split="train",
                is_serious=index % 2,
            )
        )
        raw_amount, raw_unit = (
            ("1000", "µG")
            if index == 1
            else ((str(amount), "M.G.") if index == 2 else (str(amount), "MG"))
        )
        edge_rows.append(_edge_row(primaryid, "SINGLE", raw_amount, raw_unit, "QD"))

    cohort_rows.extend(
        [
            _cohort_row("validation-outlier", date(2024, 2, 1), split="validation"),
            _cohort_row("test-combination", date(2024, 3, 1), split="test"),
        ]
    )
    edge_rows.extend(
        [
            _edge_row("validation-outlier", "SINGLE", "1000", "MG", "QD"),
            _edge_row("test-combination", "COMBINATION", "1000", "MG", "BID"),
        ]
    )
    _write(processed / "tekarx_cohort.parquet", cohort_rows)
    _write(
        processed / "drug_dictionary.parquet",
        [
            _dictionary_row("SINGLE", 1),
            _dictionary_row("COMBINATION", 2),
            _dictionary_row("COMBINATION", 3),
        ],
    )
    _write(processed / "edges" / "report_drug.parquet", edge_rows)

    record = add_tabular_features(
        data_dir=data_dir,
        rebuild_graph=False,
        memory_limit="512MB",
    )

    cohort = pq.read_table(record.output_path)
    rows = {row["primaryid"]: row for row in cohort.to_pylist()}
    outlier = rows["validation-outlier"]
    combination = rows["test-combination"]
    assert outlier["dose_normalized_relative_available_fraction"] == pytest.approx(1.0)
    assert outlier["dose_normalized_above_train_p90_count"] == 1
    assert combination["dose_normalized_numeric_fraction"] == pytest.approx(1.0)
    assert combination["dose_normalized_scheduled_fraction"] == pytest.approx(1.0)
    assert combination["dose_normalized_relative_available_fraction"] == pytest.approx(0.0)
    assert combination["dose_normalized_daily_relative_available_fraction"] == pytest.approx(0.0)
    assert combination["dose_normalized_above_train_p90_count"] == 0

    lookup = pq.read_table(processed / "dose_normalization_lookup.parquet")
    assert lookup.num_rows == 1
    lookup_row = lookup.to_pylist()[0]
    assert lookup_row["dc_id"] == 1
    assert lookup_row["support"] == 5
    assert max(
        value
        for key, value in lookup_row.items()
        if "p90" in key and value is not None
    ) <= 5

    dose_edges = pq.read_table(processed / "edges" / "report_drug_dose.parquet")
    combination_edges = dose_edges.filter(
        pc.equal(dose_edges["primaryid"], "test-combination")
    ).to_pylist()
    assert len(combination_edges) == 1
    combination_edge = combination_edges[0]
    assert combination_edge["dictionary_positive_dc_id_count"] == 2
    assert combination_edge["dc_id"] is None
    assert all(
        value is None
        for key, value in combination_edge.items()
        if "relative" in key
    )


def _cohort_row(
    primaryid: str,
    report_date: date,
    *,
    split: str,
    is_serious: int = 0,
) -> dict[str, object]:
    return {
        "primaryid": primaryid,
        "report_date": report_date,
        "age": 50,
        "age_years": 50.0,
        "sex": "F",
        "weight": "70",
        "weight_unit": "KG",
        "age_missing": 0,
        "sex_unknown": 0,
        "drug_list_str": "SINGLE",
        "num_drugs": 1,
        "is_serious": is_serious,
        "split": split,
    }


def _edge_row(
    primaryid: str,
    drugname: str,
    amount: str,
    unit: str,
    frequency: str,
) -> dict[str, object]:
    return {
        "primaryid": primaryid,
        "drugname": drugname,
        "route": "INTRAVENOUS",
        "dose_amt": amount,
        "dose_unit": unit,
        "dose_form": "INJECTION",
        "dose_freq": frequency,
    }


def _dictionary_row(faers_raw: str, dc_id: int) -> dict[str, object]:
    return {
        "faers_raw": faers_raw,
        "dc_id": dc_id,
        "atc_code": "N01",
        "ror": 1.0,
        "has_boxed_warning": 0,
    }
