"""Conservative FAERS dosage-unit and schedule normalization.

Only physically compatible units share a dimension.  The helpers in this
module also drive the SQL implementation in :mod:`tabular_features`, keeping
the small, testable Python API and the production DuckDB transform aligned.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

DOSE_NORMALIZATION_SCOPE = (
    "drug-and-dimension robust log-dose references fitted on training exposures "
    "only; held-out exposures use the frozen references"
)
DRUG_DOSE_MIN_SUPPORT = 20
RELATIVE_LOG_DOSE_CLIP = 10.0

DOSAGE_PATIENT_FEATURES = (
    "dose_normalized_numeric_fraction",
    "dose_normalized_scheduled_fraction",
    "dose_normalized_relative_available_fraction",
    "dose_normalized_relative_log_mean",
    "dose_normalized_relative_log_max",
    "dose_normalized_daily_relative_available_fraction",
    "dose_normalized_daily_relative_log_mean",
    "dose_normalized_daily_relative_log_max",
    "dose_normalized_above_train_p90_count",
    "dose_normalized_above_train_p90_fraction",
    "dose_normalized_num_parenteral_drugs",
    "dose_normalized_parenteral_fraction",
    "dose_normalized_has_iv",
    "dose_normalized_has_sc",
    "dose_normalized_has_im",
)


@dataclass(frozen=True)
class CanonicalDose:
    """A positive dose expressed in the base unit for one physical dimension."""

    dimension: str
    amount: float


def _rules() -> dict[str, tuple[str, float]]:
    rules: dict[str, tuple[str, float]] = {}

    def add(dimension: str, factor: float, *units: str) -> None:
        for unit in units:
            normalized = _normalize_unit_token(unit)
            if normalized in rules:
                raise RuntimeError(f"duplicate dosage-unit rule: {normalized}")
            rules[normalized] = (dimension, factor)

    add("mass_mg", 1e-9, "PG", "PICOGRAM", "PICOGRAMS")
    add("mass_mg", 1e-6, "NG", "NANOGRAM", "NANOGRAMS")
    add("mass_mg", 1e-3, "UG", "MCG", "MICROGRAM", "MICROGRAMS")
    add("mass_mg", 1.0, "MG", "MILLIGRAM", "MILLIGRAMS")
    add("mass_mg", 1e3, "G", "GM", "GRAM", "GRAMS")
    add("mass_mg", 1e6, "KG", "KILOGRAM", "KILOGRAMS")

    add("mass_mg_per_kg", 1e-3, "UG/KG", "MCG/KG")
    add("mass_mg_per_kg", 1.0, "MG/KG")
    add("mass_mg_per_kg", 1e3, "G/KG", "GM/KG")

    add("mass_mg_per_m2", 1e-3, "UG/M2", "MCG/M2")
    add("mass_mg_per_m2", 1.0, "MG/M2")
    add("mass_mg_per_m2", 1e3, "G/M2", "GM/M2")

    add("volume_ml", 1e-3, "UL", "MICROLITER", "MICROLITERS")
    add("volume_ml", 1.0, "ML", "CC", "MILLILITER", "MILLILITERS")
    add("volume_ml", 1e3, "L", "LITER", "LITERS")

    add("activity_iu", 1.0, "IU")
    add("activity_iu", 1e3, "KIU")
    add("activity_iu", 1e6, "MIU")
    add("activity_iu_per_kg", 1.0, "IU/KG")

    add("equivalent_meq", 1.0, "MEQ")
    add("substance_mmol", 1e-3, "UMOL")
    add("substance_mmol", 1.0, "MMOL")
    add("substance_mmol", 1e3, "MOL")

    add("radioactivity_mbq", 1e-6, "BQ")
    add("radioactivity_mbq", 1e-3, "KBQ")
    add("radioactivity_mbq", 1.0, "MBQ")
    add("radioactivity_mbq", 1e3, "GBQ")
    add("radioactivity_mbq", 37_000.0, "CI")
    add("radioactivity_mbq", 37.0, "MCI")
    add("radioactivity_mbq", 0.037, "UCI")
    add("radioactivity_mbq", 0.000037, "NCI")
    return rules


def _normalize_unit_token(value: object) -> str:
    token = str(value or "").strip()
    token = token.replace("µ", "U").replace("μ", "U")
    token = token.upper()
    token = token.replace("²", "2").replace("**", "").replace("^", "")
    return re.sub(r"[\s.]", "", token)


DOSE_UNIT_RULES = _rules()

FREQUENCY_PER_DAY: dict[str, float] = {
    "DAILY": 1.0,
    "QD": 1.0,
    "QAM": 1.0,
    "HS": 1.0,
    "BID": 2.0,
    "TID": 3.0,
    "QID": 4.0,
    "Q12H": 2.0,
    "Q8H": 3.0,
    "Q6H": 4.0,
    "Q5H": 4.8,
    "Q4H": 6.0,
    "Q3H": 8.0,
    "Q2H": 12.0,
    "QH": 24.0,
    "QOD": 0.5,
    "QW": 1.0 / 7.0,
    "/WK": 1.0 / 7.0,
    "TIW": 3.0 / 7.0,
    "Q3W": 1.0 / 21.0,
    "QOW": 1.0 / 14.0,
    "QM": 1.0 / (365.25 / 12.0),
    "/MONTH": 1.0 / (365.25 / 12.0),
    "/YR": 1.0 / 365.25,
    "1/YR": 1.0 / 365.25,
}


def canonicalize_dose(amount: object, unit: object) -> CanonicalDose | None:
    """Return a positive finite canonical dose, or ``None`` when unsafe.

    Counts, percentages and unrecognized units deliberately remain missing;
    converting those values would require product strength or concentration.
    """
    try:
        numeric = float(amount)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    rule = DOSE_UNIT_RULES.get(_normalize_unit_token(unit))
    if rule is None:
        return None
    dimension, factor = rule
    canonical = numeric * factor
    if not math.isfinite(canonical) or canonical <= 0:
        return None
    return CanonicalDose(dimension=dimension, amount=canonical)


def frequency_per_day(value: object) -> float | None:
    """Convert an exact, unambiguous FAERS frequency code to events per day."""
    token = re.sub(r"\s+", "", str(value or "").strip().upper())
    return FREQUENCY_PER_DAY.get(token)


def normalized_unit_sql(expression: str) -> str:
    """Return DuckDB SQL equivalent to :func:`_normalize_unit_token`."""
    return (
        "replace(replace(replace("
        f"regexp_replace(upper(replace(replace(trim(coalesce({expression}, '')), "
        "'\u00b5', 'U'), '\u03bc', 'U')), '[[:space:].]', '', 'g'), "
        "'\u00b2', '2'), '**', ''), '^', '')"
    )


def normalized_frequency_sql(expression: str) -> str:
    """Return DuckDB SQL equivalent to the schedule-token normalization."""
    return (
        f"regexp_replace(upper(trim(coalesce({expression}, ''))), "
        "'[[:space:]]', '', 'g')"
    )


def dose_dimension_case_sql(unit_expression: str) -> str:
    """Build the canonical-dimension CASE expression from ``DOSE_UNIT_RULES``."""
    grouped: dict[str, list[str]] = {}
    for unit, (dimension, _factor) in DOSE_UNIT_RULES.items():
        grouped.setdefault(dimension, []).append(unit)
    branches = " ".join(
        f"WHEN {unit_expression} IN ({_sql_strings(units)}) THEN '{dimension}'"
        for dimension, units in sorted(grouped.items())
    )
    return f"CASE {branches} END"


def dose_factor_case_sql(unit_expression: str) -> str:
    """Build the canonical multiplier CASE expression from ``DOSE_UNIT_RULES``."""
    grouped: dict[float, list[str]] = {}
    for unit, (_dimension, factor) in DOSE_UNIT_RULES.items():
        grouped.setdefault(factor, []).append(unit)
    branches = " ".join(
        f"WHEN {unit_expression} IN ({_sql_strings(units)}) THEN {factor!r}"
        for factor, units in sorted(grouped.items())
    )
    return f"CASE {branches} END"


def frequency_case_sql(frequency_expression: str) -> str:
    """Build the safe schedule CASE expression from ``FREQUENCY_PER_DAY``."""
    grouped: dict[float, list[str]] = {}
    for frequency, per_day in FREQUENCY_PER_DAY.items():
        grouped.setdefault(per_day, []).append(frequency)
    branches = " ".join(
        f"WHEN {frequency_expression} IN ({_sql_strings(values)}) THEN {per_day!r}"
        for per_day, values in sorted(grouped.items())
    )
    return f"CASE {branches} END"


def _sql_strings(values: list[str]) -> str:
    return ", ".join(f"'{value.replace(chr(39), chr(39) * 2)}'" for value in values)
