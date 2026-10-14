"""Shared model logic for the TekaRx Streamlit app.

This module is imported by every page. It contains artifact loading,
medication matching, feature engineering, prediction, and threshold
retrieval. **No UI code lives here.**

All functions are copied verbatim from the original single-file
``streamlit_app.py`` to preserve scoring outputs exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

# The frozen model is CPU-bound. Limiting numerical worker pools avoids
# duplicate thread-local buffers and makes latency/memory predictable during a
# presentation. These must be set before NumPy/scikit-learn import.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ.setdefault(_name, "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import shap
except ImportError:  # The UI can still render non-explainability pages.
    shap = None
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

    serialized_bundle = joblib.load(status.bundle_path)
    # The visitor UI only serves the frozen Random Forest. Drop the unused
    # logistic, SVM, and AdaBoost estimators immediately so they do not remain
    # resident in the Streamlit process.
    required_bundle_keys = (MODEL_KEY, "imputer", "scaler", "feature_cols", "seed")
    bundle = {
        key: serialized_bundle[key]
        for key in required_bundle_keys
        if key in serialized_bundle
    }
    del serialized_bundle
    _validate_bundle(bundle)
    dictionary = pd.read_parquet(status.dictionary_path)
    _validate_dictionary(dictionary)
    return {
        "bundle": bundle,
        "dictionary": dictionary,
        # Materialize this once. Building/sorting 92k selectbox options on
        # every Streamlit rerun is a major source of UI latency.
        "drug_choices": _drug_choices(dictionary),
    }


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


_DRUG_CHOICES_CACHE: dict[int, tuple[str, ...]] = {}


def _drug_choices(dictionary: pd.DataFrame) -> tuple[str, ...]:
    """Return stable, deduplicated medication choices for this dictionary."""
    cache_key = id(dictionary)
    choices = _DRUG_CHOICES_CACHE.get(cache_key)
    if choices is None:
        choices = tuple(
            sorted(
                dictionary["faers_raw"]
                .dropna()
                .drop_duplicates()
                .astype(str)
                .tolist()
            )
        )
        _DRUG_CHOICES_CACHE[cache_key] = choices
    return choices


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
    choices = _drug_choices(dictionary)
    query_upper = query.strip().upper()
    candidates = process.extract(
        query_upper, choices, scorer=fuzz.partial_ratio, limit=limit
    )
    return [candidate[0] for candidate in candidates if candidate[1] >= 70]


def match_medications(
    queries: list[str], dictionary: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Match medication queries against drug dictionary."""
    choices = _drug_choices(dictionary)
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


# =====================================================================
# SHAP Explainability & Chart Generation
# =====================================================================

_SHAP_EXPLAINER_CACHE: dict[int, Any] = {}


def _get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Retrieve or initialize a cached TreeExplainer for the given tree model."""
    if shap is None:
        raise RuntimeError(
            "SHAP is not installed. Install frontend/requirements.txt before viewing explanations."
        )
    model_id = id(model)
    if model_id not in _SHAP_EXPLAINER_CACHE:
        _SHAP_EXPLAINER_CACHE[model_id] = shap.TreeExplainer(model)
    return _SHAP_EXPLAINER_CACHE[model_id]


def compute_shap(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Compute local Tree SHAP values explaining the positive class prediction."""
    feature_cols = bundle["feature_cols"]
    aligned = frame.reindex(columns=feature_cols)
    transformed = bundle["scaler"].transform(
        bundle["imputer"].transform(aligned)
    )

    model = bundle[MODEL_KEY]
    if shap is None:
        # Keep the UI usable in a minimal install.  This is a clearly marked
        # approximation (global tree importance × local standardized value);
        # the normal path below is exact Tree SHAP whenever SHAP is installed.
        importance = np.asarray(
            getattr(model, "feature_importances_", np.ones(len(feature_cols))),
            dtype=float,
        )
        shap_vals = importance * np.asarray(transformed[0], dtype=float)
        feature_values = np.asarray(transformed[0], dtype=float)
        return shap_vals, list(feature_cols), feature_values

    explainer = _get_tree_explainer(model)
    sv = explainer.shap_values(transformed)

    if isinstance(sv, list):
        shap_vals = (
            sv[1][0]
            if len(sv) > 1
            else (sv[0][0] if sv[0].ndim > 1 else sv[0])
        )
    elif isinstance(sv, np.ndarray):
        if sv.ndim == 3:
            shap_vals = (
                sv[0, :, 1]
                if sv.shape[2] > 1
                else sv[0, :, 0]
            )
        elif sv.ndim == 2:
            shap_vals = sv[0]
        else:
            shap_vals = sv
    else:
        shap_vals = np.asarray(sv).ravel()

    shap_vals = np.asarray(shap_vals, dtype=float)
    feature_values = np.asarray(transformed[0], dtype=float)
    feature_names = list(feature_cols)

    return shap_vals, feature_names, feature_values


def render_shap_chart(
    shap_values: np.ndarray | list[float],
    feature_names: list[str],
    feature_values: np.ndarray | list[float],
    max_display: int = 5,
) -> plt.Figure:
    """Render an authentic TekaRx clinical-themed horizontal bar chart of top SHAP contributors."""
    labels = [_feature_label(f) for f in feature_names]
    df = pd.DataFrame({
        "feature": labels,
        "shap": np.asarray(shap_values, dtype=float),
        "value": np.asarray(feature_values, dtype=float),
    })
    df["abs_shap"] = np.abs(df["shap"])
    top = df.sort_values("abs_shap", ascending=True).tail(max_display)

    GREEN = "#0A7A55"
    AMBER = "#D9770A"
    BG_CARD = "#FFFFFF"
    TEXT = "#10231C"
    MUTED = "#53645C"
    BORDER = "#DDE5E1"

    colors = [AMBER if v > 0 else GREEN for v in top["shap"]]

    fig, ax = plt.subplots(
        figsize=(8.5, max_display * 0.68 + 1.2), facecolor=BG_CARD
    )
    ax.set_facecolor(BG_CARD)

    bars = ax.barh(
        top["feature"],
        top["shap"],
        color=colors,
        edgecolor="#DDE5E1",
        linewidth=1.0,
        height=0.6,
        zorder=3,
    )

    max_abs = max(float(top["abs_shap"].max()), 0.05)
    pad_dist = max_abs * 0.05
    for bar in bars:
        w = bar.get_width()
        x_pos = w + (pad_dist if w >= 0 else -pad_dist)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{w:+.3f}",
            ha="left" if w >= 0 else "right",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=TEXT,
            zorder=4,
        )

    ytick_labels = [
        f"{r['feature']}  (scaled: {r['value']:+.2f})" for _, r in top.iterrows()
    ]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        ytick_labels, fontsize=10, color=TEXT, fontweight="medium"
    )

    min_val = min(float(top["shap"].min()), 0.0)
    max_val = max(float(top["shap"].max()), 0.0)
    x_span = max(max_val - min_val, 0.08)
    ax.set_xlim(min_val - x_span * 0.22, max_val + x_span * 0.22)

    ax.axvline(0, color="#AABBB2", linewidth=1.2, linestyle="--", zorder=2)
    ax.grid(axis="x", color="#E6ECE9", linestyle=":", linewidth=0.8, zorder=1)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(BORDER)

    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", colors=TEXT, length=0)

    ax.set_title(
        "SHAP Local Explanations (Feature Impact on Priority Signal)",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
        pad=14,
        loc="left",
    )
    ax.set_xlabel(
        "← Reduces Priority (Protective)   |   Increases Priority (Risk) →",
        color=MUTED,
        fontsize=9,
        labelpad=8,
    )
    fig.tight_layout()
    return fig
