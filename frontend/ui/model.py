"""Shared model logic for the TekaRx Streamlit app.

This module is imported by every page. It contains artifact loading,
medication matching, feature engineering, prediction, and threshold
retrieval. **No UI code lives here.**

All functions are copied verbatim from the original single-file
``streamlit_app.py`` to preserve scoring outputs exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
BUNDLE_PATH = PROCESSED / "models" / "imrad_models.joblib"
DICTIONARY_PATH = PROCESSED / "drug_dictionary.parquet"
COHORT_PATH = PROCESSED / "tekarx_cohort.parquet"

DISCLAIMER = "Research decision support, not medical advice. Always exercise clinical judgment."
MODEL_KEY = "random_forest"
MODEL_NAME = "Random Forest"
DEFAULT_THRESHOLD = 0.50
MATCHED_CUTOFF = 90.0
AMBIGUOUS_CUTOFF = 75.0

BUILD_COMMANDS = (
    "python -m pip install -e .[imrad,notebook]\n"
    "python build_imrad_artifacts.py\n"
    "# Expected outputs:\n"
    "# data/processed/drug_dictionary.parquet\n"
    "# data/processed/models/imrad_models.joblib"
)

# Sample scenarios
HIGH_PRIORITY_MEDS = ["WARFARIN", "IBUPROFEN", "TRAMADOL", "METOPROLOL", "OMEPRAZOLE"]
HIGH_PRIORITY_AGE = 74
HIGH_PRIORITY_SEX = "Female"

MAINTENANCE_MEDS = ["METFORMIN", "ATORVASTATIN", "LEVOTHYROXINE"]
MAINTENANCE_AGE = 52
MAINTENANCE_SEX = "Male"


# =====================================================================
# Model Artifacts & Validation
# =====================================================================


@dataclass(frozen=True)
class ArtifactStatus:
    bundle_path: Path
    dictionary_path: Path
    missing: tuple[Path, ...]


def _artifact_status() -> ArtifactStatus:
    required = (BUNDLE_PATH, DICTIONARY_PATH)
    return ArtifactStatus(
        bundle_path=BUNDLE_PATH,
        dictionary_path=DICTIONARY_PATH,
        missing=tuple(path for path in required if not path.is_file()),
    )


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict[str, Any]:
    """Load persisted model artifacts without retraining."""
    status = _artifact_status()
    if status.missing:
        raise FileNotFoundError("Missing TekaRx artifacts")

    bundle = joblib.load(status.bundle_path)
    _validate_bundle(bundle)
    dictionary = pd.read_parquet(status.dictionary_path)
    _validate_dictionary(dictionary)
    return {"bundle": bundle, "dictionary": dictionary}


def _validate_bundle(bundle: dict[str, Any]) -> None:
    required = {"feature_cols", "imputer", "scaler", MODEL_KEY}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Model bundle is missing keys: {', '.join(missing)}")
    if not hasattr(bundle[MODEL_KEY], "predict_proba"):
        raise ValueError(
            f"Model bundle key {MODEL_KEY!r} does not expose predict_proba"
        )


def _validate_dictionary(dictionary: pd.DataFrame) -> None:
    required = {"faers_raw", "dc_id", "atc_code", "ror", "has_boxed_warning"}
    missing = sorted(required - set(dictionary.columns))
    if missing:
        raise ValueError(
            f"Drug dictionary is missing columns: {', '.join(missing)}"
        )


# =====================================================================
# Medication Parsing, Autocomplete & Matching
# =====================================================================


def parse_medications(value: str) -> list[str]:
    """Parse comma or newline separated medications into clean strings."""
    return [
        item.strip()
        for item in value.replace("\n", ",").split(",")
        if item.strip()
    ]


def generate_record(
    bundle: dict[str, Any], dictionary: pd.DataFrame
) -> tuple[float, str, list[str]]:
    """Generate a random testable record.

    Returns: (age, sex, med_list)
    """
    if COHORT_PATH.is_file():
        try:
            cohort = pd.read_parquet(COHORT_PATH)
            if len(cohort) > 0:
                record = cohort.sample(n=1).iloc[0].to_dict()
                raw_age = record.get("age")
                if raw_age is not None and pd.notna(raw_age):
                    try:
                        age = float(raw_age)
                    except (ValueError, TypeError):
                        age = float(np.random.randint(30, 80))
                else:
                    age = float(np.random.randint(30, 80))
                sex = str(record.get("sex", "Unknown"))
                meds_col = None
                for col in cohort.columns:
                    if "med" in col.lower() or "drug" in col.lower():
                        meds_col = col
                        break
                if meds_col and pd.notna(record.get(meds_col)):
                    meds = parse_medications(str(record[meds_col]))
                else:
                    sample_size = min(4, len(dictionary))
                    meds = (
                        dictionary["faers_raw"]
                        .dropna()
                        .drop_duplicates()
                        .sample(sample_size, random_state=None)
                        .tolist()
                    )
                return age, sex, meds
        except Exception:
            pass

    age = float(np.random.randint(30, 80))
    sex = str(np.random.choice(["Female", "Male", "Unknown"]))
    sample_size = min(int(np.random.randint(3, 6)), len(dictionary))
    meds = (
        dictionary["faers_raw"]
        .dropna()
        .drop_duplicates()
        .sample(sample_size, random_state=None)
        .tolist()
    )
    return age, sex, meds


def get_drug_suggestions(
    query: str, dictionary: pd.DataFrame, limit: int = 10
) -> list[str]:
    """Get top drug suggestions using RapidFuzz typeahead."""
    if not query or len(query) < 1:
        return []
    choices = (
        dictionary["faers_raw"].dropna().drop_duplicates().astype(str).tolist()
    )
    query_upper = query.strip().upper()
    candidates = process.extract(
        query_upper, choices, scorer=fuzz.partial_ratio, limit=limit
    )
    return [candidate[0] for candidate in candidates if candidate[1] >= 70]


def match_medications(
    queries: list[str], dictionary: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Match medication queries against drug dictionary."""
    choices = (
        dictionary["faers_raw"].dropna().drop_duplicates().astype(str).tolist()
    )
    rows: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []

    for query in queries:
        query_normalized = query.strip().upper()
        candidates = process.extract(
            query_normalized, choices, scorer=fuzz.WRatio, limit=3
        )
        if not candidates:
            matches.append(
                {
                    "input": query,
                    "status": "unmatched",
                    "match": "",
                    "score": 0.0,
                    "suggestions": "No match found",
                    "used": False,
                }
            )
            continue

        match, score, _ = candidates[0]
        status = _match_status(float(score))
        used = status == "matched"
        suggestions = ", ".join(str(c[0]) for c in candidates[:3])
        if used:
            rows.extend(
                dictionary.loc[dictionary["faers_raw"] == match].to_dict(
                    "records"
                )
            )
        matches.append(
            {
                "input": query,
                "status": status,
                "match": str(match) if used else "",
                "score": float(score),
                "suggestions": suggestions,
                "used": used,
            }
        )

    return pd.DataFrame(rows), matches


def _match_status(score: float) -> str:
    if score >= MATCHED_CUTOFF:
        return "matched"
    if score >= AMBIGUOUS_CUTOFF:
        return "ambiguous"
    return "unmatched"


# =====================================================================
# Feature Engineering & Model Inference
# =====================================================================


def _feature_defaults(bundle: dict[str, Any]) -> dict[str, float]:
    values = bundle["imputer"].statistics_
    return dict(zip(bundle["feature_cols"], values, strict=True))


def build_features(
    age: float,
    sex: str,
    matched: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build feature frame from matched drugs and patient demographics."""
    columns = bundle["feature_cols"]
    values = _feature_defaults(bundle)
    age = float(age)
    values.update(
        {
            "age_group_0_17": float(age < 18),
            "age_group_18_40": float(18 <= age <= 40),
            "age_group_41_64": float(41 <= age <= 64),
            "age_group_65_plus": float(age >= 65),
            "age_imputed_years": age,
            "age_missing": 0.0,
            "sex_unknown": float(sex == "Unknown"),
        }
    )

    if matched.empty:
        rors = np.array([1.0])
        atc_codes: list[str] = []
        boxed = 0.0
    else:
        rors = (
            pd.to_numeric(matched["ror"], errors="coerce")
            .fillna(1.0)
            .to_numpy()
        )
        atc_codes = [
            code.strip().upper()
            for raw in matched["atc_code"].dropna().astype(str)
            for code in raw.split("|")
            if len(code.strip()) >= 3 and code.strip()[0].isalpha()
        ]
        boxed = float(
            pd.to_numeric(matched["has_boxed_warning"], errors="coerce")
            .fillna(0)
            .max()
        )

    unique_atc = sorted(set(atc_codes))
    l1 = [code[0] for code in unique_atc]
    l2 = [code[:3] for code in unique_atc]
    l3 = [code[:4] for code in unique_atc if len(code) >= 4]
    l4 = [code[:5] for code in unique_atc if len(code) >= 5]
    num_drugs = int(len(matched))
    high_risk = [code for code in unique_atc if code[0] in {"N", "B", "C", "M"}]

    values.update(
        {
            "atc_diversity": float(len(set(l1))),
            "atc_l2_diversity": float(len(set(l2))),
            "atc_l3_diversity": float(len(set(l3))),
            "atc_l4_diversity": float(len(set(l4))),
            "max_ror": float(np.max(rors)),
            "mean_log_ror": float(np.log(np.maximum(rors, 1e-12)).mean()),
            "high_ror_count": float(np.sum(rors > 2.0)),
            "has_boxed_warning": boxed,
            "num_drugs": float(num_drugs),
            "num_drugs_squared": float(num_drugs**2),
            "num_high_risk_atc": float(len(set(high_risk))),
            "num_high_risk_atc_groups": float(
                sum(
                    code[:4] in {"N02A", "B01A", "M01A", "A10A"}
                    for code in unique_atc
                )
            ),
            "polypharmacy_age": float(num_drugs * age),
            "therapeutic_duplicates": float(max(len(l1) - len(set(l1)), 0)),
            "therapeutic_duplicates_l2": float(max(len(l2) - len(set(l2)), 0)),
        }
    )
    for letter in "abcdefghijklmnopqrstuvwxyz":
        key = f"atc_l1_count_{letter}"
        if key in columns:
            values[key] = float(l1.count(letter.upper()))

    frame = pd.DataFrame(
        [{column: values.get(column, np.nan) for column in columns}],
        columns=columns,
    )
    summary = {
        "Matched medications": num_drugs,
        "Highest ROR": values["max_ror"],
        "Average ROR": float(np.exp(values["mean_log_ror"])),
        "High-ROR count": int(values["high_ror_count"]),
        "ATC diversity": int(values["atc_diversity"]),
        "High-priority ATC": int(values["num_high_risk_atc"]),
        "Boxed warning": "Yes" if boxed > 0 else "No",
    }
    return frame, summary


def predict(frame: pd.DataFrame, artifacts: dict[str, Any]) -> float:
    """Run model prediction with frozen scikit-learn pipeline."""
    bundle = artifacts["bundle"]
    aligned = frame.reindex(columns=bundle["feature_cols"])
    transformed = bundle["scaler"].transform(
        bundle["imputer"].transform(aligned)
    )
    return float(bundle[MODEL_KEY].predict_proba(transformed)[0, 1])


def threshold(bundle: dict[str, Any]) -> float:
    """Retrieve frozen operating threshold."""
    for key in ("threshold", "frozen_threshold", "operating_threshold"):
        value = bundle.get(key)
        if isinstance(value, int | float):
            return float(value)
    thresholds = bundle.get("thresholds")
    if isinstance(thresholds, dict):
        for key in (MODEL_KEY, "default"):
            value = thresholds.get(key)
            if isinstance(value, int | float):
                return float(value)
    return DEFAULT_THRESHOLD


def top_contributors(
    bundle: dict[str, Any], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    """Extract top 5 contributing features as a list of dicts.

    Each dict has keys: Feature, Value, Importance, Contribution.
    """
    model = bundle[MODEL_KEY]
    feature_cols = bundle["feature_cols"]
    if hasattr(model, "feature_importances_"):
        strength = np.abs(model.feature_importances_)
    else:
        strength = np.zeros(len(feature_cols))

    raw_values = frame.iloc[0].to_dict()
    records = []
    for col, imp in zip(feature_cols, strength, strict=True):
        val = raw_values.get(col)
        records.append(
            {
                "Feature": _feature_label(col),
                "Value": _format_feature_value(col, val),
                "Importance": float(imp),
            }
        )

    import pandas as _pd

    df = (
        _pd.DataFrame(records)
        .sort_values("Importance", ascending=False)
        .head(5)
    )
    total_imp = df["Importance"].sum()
    if total_imp > 0:
        df["Contribution"] = df["Importance"].map(
            lambda v: v / total_imp * 100.0
        )
    else:
        df["Contribution"] = 0.0

    return df.to_dict("records")


def _feature_label(name: str) -> str:
    """Translate raw feature identifiers into clinical labels."""
    labels = {
        "max_ror": "Highest drug ROR",
        "mean_log_ror": "Average log ROR",
        "high_ror_count": "High-ROR count",
        "has_boxed_warning": "Boxed warning",
        "num_drugs": "Drug count",
        "num_drugs_squared": "Drug count\u00b2",
        "polypharmacy_age": "Polypharmacy \u00d7 Age",
        "atc_diversity": "ATC diversity",
        "atc_l2_diversity": "ATC L2 diversity",
        "num_high_risk_atc": "High-risk ATC",
        "num_high_risk_atc_groups": "High-risk groups",
        "therapeutic_duplicates": "Therapeutic duplicates",
        "age_imputed_years": "Patient age",
        "age_group_65_plus": "Age \u226565",
        "sex_unknown": "Sex unspecified",
    }
    if name.startswith("atc_l1_count_"):
        letter = name.split("_")[-1].upper()
        return f"ATC class {letter}"
    return labels.get(name, name.replace("_", " ").title())


def _format_feature_value(name: str, val: Any) -> str:
    """Format feature values with clear clinical precision."""
    if val is None or pd.isna(val):
        return "N/A"
    val = float(val)
    if "has_" in name or "_missing" in name or name.startswith("age_group_"):
        return "Yes" if val >= 0.5 else "No"
    if name in {
        "num_drugs",
        "high_ror_count",
        "atc_diversity",
        "atc_l2_diversity",
        "therapeutic_duplicates",
    }:
        return f"{int(round(val))}"
    if name == "age_imputed_years":
        return f"{val:.0f} yr"
    if "ror" in name:
        return f"{val:.2f}"
    return f"{val:.2f}"
