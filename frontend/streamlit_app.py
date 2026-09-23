"""Visitor-facing Streamlit dashboard for the frozen TekaRx IMRAD models.

Design Read:
Reading this as: clinical decision-support SaaS tool for clinicians and researchers,
in an authentic dark clinical SaaS dashboard style, dial ENERGY 2 / RHYTHM 2 / MOTION 1.

Key Purpose Decisions:
- Palette: Dark clinical neutral surface (#0b0f0d, #131916) with clinical emerald (#10b981)
  as primary brand accent; amber/red reserved strictly for clinical risk.
- Typography: System sans with tabular numerals enabled to ensure numerical safety alignment.
- Spacing: Strict 8pt spatial rhythm (8px, 16px, 24px, 32px) for structured clinical hierarchy.
- Radii: 6px for small elements/inputs, 10px for cards to prevent generic pill distortion.
- Elevation: Two deliberate shadow tiers (resting card vs. elevated score panel).
- Icons: Specific medical/data SVGs (shield, capsule, warning, pulse) tied directly to metrics.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
BUNDLE_PATH = PROCESSED / "models" / "imrad_models.joblib"
DICTIONARY_PATH = PROCESSED / "drug_dictionary.parquet"
COHORT_PATH = PROCESSED / "tekarx_cohort.parquet"

DISCLAIMER = "Research decision-support, not medical advice."
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


def _parse_medications(value: str) -> list[str]:
    """Parse comma or newline separated medications into clean strings."""
    return [
        item.strip()
        for item in value.replace("\n", ",").split(",")
        if item.strip()
    ]


def _generate_record(
    bundle: dict[str, Any], dictionary: pd.DataFrame
) -> tuple[float, str, str, list[str]]:
    """Generate a testable record from cohort or synthesized statistics.

    Returns: (age, sex, meds_str, generated_field_names)
    """
    if COHORT_PATH.is_file():
        try:
            cohort = pd.read_parquet(COHORT_PATH)
            if len(cohort) > 0:
                record = cohort.sample(n=1).iloc[0].to_dict()
                age = float(record.get("age", np.random.randint(30, 80)))
                sex = str(record.get("sex", "Unknown"))
                meds_col = None
                for col in cohort.columns:
                    if "med" in col.lower() or "drug" in col.lower():
                        meds_col = col
                        break

                if meds_col and pd.notna(record.get(meds_col)):
                    meds_str = str(record[meds_col])
                else:
                    sample_size = min(4, len(dictionary))
                    frequent = (
                        dictionary["faers_raw"]
                        .dropna()
                        .drop_duplicates()
                        .sample(sample_size, random_state=None)
                        .tolist()
                    )
                    meds_str = "\n".join(frequent)

                return age, sex, meds_str, ["age", "sex", "medications"]
        except Exception:
            pass

    age = float(np.random.randint(30, 80))
    sex = str(np.random.choice(["Female", "Male", "Unknown"]))
    sample_size = min(int(np.random.randint(3, 6)), len(dictionary))
    frequent = (
        dictionary["faers_raw"]
        .dropna()
        .drop_duplicates()
        .sample(sample_size, random_state=None)
        .tolist()
    )
    meds_str = "\n".join(frequent)
    return age, sex, meds_str, ["age", "sex", "medications"]


def _get_drug_suggestions(
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
    return [
        candidate[0] for candidate in candidates if candidate[1] >= 70
    ]


def _match_medications(
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


def _render_match_feedback(match_details: list[dict[str, Any]]) -> int:
    """Return count of successfully matched medications for test compatibility."""
    if not match_details:
        return 0
    return sum(1 for m in match_details if m.get("used", False))


# =====================================================================
# Feature Engineering & Model Inference
# =====================================================================


def _feature_defaults(bundle: dict[str, Any]) -> dict[str, float]:
    values = bundle["imputer"].statistics_
    return dict(zip(bundle["feature_cols"], values, strict=True))


def _build_features(
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


def _predict(frame: pd.DataFrame, artifacts: dict[str, Any]) -> float:
    """Run model prediction with frozen scikit-learn pipeline."""
    bundle = artifacts["bundle"]
    aligned = frame.reindex(columns=bundle["feature_cols"])
    transformed = bundle["scaler"].transform(
        bundle["imputer"].transform(aligned)
    )
    return float(bundle[MODEL_KEY].predict_proba(transformed)[0, 1])


def _threshold(bundle: dict[str, Any]) -> float:
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


def _top_contributors(
    bundle: dict[str, Any], frame: pd.DataFrame
) -> pd.DataFrame:
    """Extract top 5 contributing features."""
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

    df = (
        pd.DataFrame(records)
        .sort_values("Importance", ascending=False)
        .head(5)
    )
    total_imp = df["Importance"].sum()
    if total_imp > 0:
        df["Contribution"] = df["Importance"].map(
            lambda v: f"{v / total_imp:.1%}"
        )
    else:
        df["Contribution"] = "N/A"
    return df[["Feature", "Value", "Contribution"]]


def _feature_label(name: str) -> str:
    """Translate raw feature identifiers into clinical labels."""
    labels = {
        "max_ror": "Highest Drug ROR",
        "mean_log_ror": "Average Log ROR",
        "high_ror_count": "High-ROR Count",
        "has_boxed_warning": "Boxed Warning",
        "num_drugs": "Drug Count",
        "num_drugs_squared": "Drug Count²",
        "polypharmacy_age": "Polypharmacy × Age",
        "atc_diversity": "ATC Diversity",
        "atc_l2_diversity": "ATC L2 Diversity",
        "num_high_risk_atc": "High-Risk ATC",
        "num_high_risk_atc_groups": "High-Risk Groups",
        "therapeutic_duplicates": "Therapeutic Duplicates",
        "age_imputed_years": "Patient Age",
        "age_group_65_plus": "Age ≥65",
        "sex_unknown": "Sex Unspecified",
    }
    if name.startswith("atc_l1_count_"):
        letter = name.split("_")[-1].upper()
        return f"ATC Class {letter}"
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
# Design System & Global Stylesheet
# =====================================================================


def _render_css() -> None:
    """Inject cohesive clinical SaaS design system CSS into Streamlit."""
    st.markdown(
        """
        <style>
        :root {
          /* Color tokens */
          --bg-base: #0b0f0d;
          --bg-card: #131916;
          --bg-elevated: #18221d;
          --bg-subtle: #0f1512;
          --border-subtle: rgba(255, 255, 255, 0.08);
          --border-strong: #24352b;
          --border-focus: #10b981;

          /* Text tokens */
          --text-primary: #f1f5f3;
          --text-secondary: #9cb0a4;
          --text-muted: #72857a;

          /* Accents */
          --accent-green: #10b981;
          --accent-green-hover: #059669;
          --accent-green-tint: rgba(16, 185, 129, 0.12);
          --accent-green-border: rgba(16, 185, 129, 0.35);

          /* Status colors (strictly for outcomes) */
          --status-red: #f87171;
          --status-red-bg: rgba(239, 68, 68, 0.12);
          --status-red-border: rgba(239, 68, 68, 0.35);

          --status-amber: #fbbf24;
          --status-amber-bg: rgba(245, 158, 11, 0.12);
          --status-amber-border: rgba(245, 158, 11, 0.35);

          --status-green: #34d399;
          --status-green-bg: rgba(16, 185, 129, 0.12);
          --status-green-border: rgba(16, 185, 129, 0.35);

          /* Spacing tokens (8pt grid) */
          --space-1: 4px;
          --space-2: 8px;
          --space-3: 16px;
          --space-4: 24px;
          --space-5: 32px;

          /* Radii (strictly consistent, no pill slop) */
          --radius-sm: 6px;
          --radius-md: 10px;

          /* Elevation shadows */
          --shadow-resting: 0 2px 8px rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.25);
          --shadow-elevated: 0 8px 24px rgba(0, 0, 0, 0.55), 0 2px 6px rgba(0, 0, 0, 0.3);
        }

        /* Base Application Resets */
        html, body, [data-testid="stAppViewContainer"] {
          background-color: var(--bg-base) !important;
          color: var(--text-primary) !important;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
          font-variant-numeric: tabular-nums;
        }

        .main .block-container {
          max-width: 1240px;
          padding-top: var(--space-4);
          padding-bottom: var(--space-5);
          padding-left: var(--space-4);
          padding-right: var(--space-4);
        }

        /* Top Branded Bar */
        .saas-topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-bottom: var(--space-3);
          margin-bottom: var(--space-4);
          border-bottom: 1px solid var(--border-subtle);
        }

        .saas-brand {
          display: flex;
          align-items: center;
          gap: var(--space-3);
        }

        .saas-logo-wrap {
          width: 40px;
          height: 40px;
          border-radius: var(--radius-sm);
          background: var(--bg-elevated);
          border: 1px solid var(--accent-green-border);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--accent-green);
        }

        .saas-brand-title {
          font-size: 1.35rem;
          font-weight: 700;
          letter-spacing: -0.02em;
          color: var(--text-primary);
          margin: 0;
          line-height: 1.2;
        }

        .saas-brand-subtitle {
          font-size: 0.82rem;
          color: var(--text-secondary);
          margin: 0;
          letter-spacing: 0.01em;
        }

        .saas-badges {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        .saas-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 10px;
          border-radius: var(--radius-sm);
          font-size: 0.75rem;
          font-weight: 500;
          background: var(--bg-card);
          border: 1px solid var(--border-subtle);
          color: var(--text-secondary);
        }

        .saas-badge-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--accent-green);
        }

        /* Global Tab Overrides */
        .stTabs [data-baseweb="tab-list"] {
          background: transparent !important;
          border-bottom: 1px solid var(--border-subtle) !important;
          gap: var(--space-2) !important;
          margin-bottom: var(--space-4) !important;
        }

        .stTabs [data-baseweb="tab"] {
          background: transparent !important;
          color: var(--text-secondary) !important;
          border-bottom: 2px solid transparent !important;
          font-size: 0.9rem !important;
          font-weight: 500 !important;
          padding: 10px 16px !important;
          transition: all 0.2s ease !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
          color: var(--text-primary) !important;
        }

        .stTabs [aria-selected="true"] {
          color: var(--accent-green) !important;
          border-bottom-color: var(--accent-green) !important;
          font-weight: 600 !important;
        }

        /* Elevated Card Architecture */
        .saas-card {
          background: var(--bg-card);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-md);
          padding: var(--space-4);
          box-shadow: var(--shadow-resting);
          margin-bottom: var(--space-3);
          transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .saas-card--elevated {
          background: var(--bg-elevated);
          border-color: var(--border-strong);
          box-shadow: var(--shadow-elevated);
        }

        .saas-card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-3);
        }

        .saas-card-title {
          font-size: 0.95rem;
          font-weight: 600;
          color: var(--text-primary);
          margin: 0;
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        /* Input Overrides */
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] {
          background-color: var(--bg-elevated) !important;
          border: 1px solid var(--border-strong) !important;
          border-radius: var(--radius-sm) !important;
          color: var(--text-primary) !important;
          transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }

        div[data-baseweb="input"]:hover,
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="textarea"]:hover {
          border-color: #3b5043 !important;
        }

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="select"]:focus-within > div,
        div[data-baseweb="textarea"]:focus-within {
          border-color: var(--accent-green) !important;
          box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25) !important;
        }

        label[data-testid="stWidgetLabel"] p {
          color: var(--text-secondary) !important;
          font-size: 0.84rem !important;
          font-weight: 500 !important;
          margin-bottom: 4px !important;
        }

        /* Button Styling Overrides */
        .stButton > button {
          border-radius: var(--radius-sm) !important;
          font-size: 0.88rem !important;
          font-weight: 600 !important;
          padding: 8px 16px !important;
          border: 1px solid var(--border-strong) !important;
          background: var(--bg-elevated) !important;
          color: var(--text-primary) !important;
          transition: all 0.18s ease !important;
          box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
        }

        .stButton > button:hover {
          border-color: var(--accent-green-border) !important;
          background: #202d26 !important;
          color: #ffffff !important;
        }

        .stButton > button:focus {
          outline: none !important;
          border-color: var(--accent-green) !important;
          box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.3) !important;
        }

        .stButton > button[kind="primary"] {
          background: var(--accent-green) !important;
          border-color: var(--accent-green) !important;
          color: #062215 !important;
          font-weight: 700 !important;
        }

        .stButton > button[kind="primary"]:hover {
          background: var(--accent-green-hover) !important;
          border-color: var(--accent-green-hover) !important;
          color: #ffffff !important;
        }

        /* Preset Action Bar */
        .preset-bar {
          display: flex;
          gap: var(--space-2);
          margin-bottom: var(--space-3);
          flex-wrap: wrap;
        }

        /* Medication Chips */
        .med-chips-container {
          display: flex;
          flex-wrap: wrap;
          gap: var(--space-2);
          margin-top: var(--space-2);
          margin-bottom: var(--space-2);
        }

        .med-chip {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: var(--bg-elevated);
          border: 1px solid var(--border-strong);
          color: var(--text-primary);
          border-radius: var(--radius-sm);
          padding: 4px 10px;
          font-size: 0.8rem;
          font-weight: 500;
        }

        /* Score Gauge Display */
        .score-hero {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: var(--space-4) var(--space-3);
        }

        .gauge-svg {
          width: 170px;
          height: 170px;
          transform: rotate(-90deg);
        }

        .gauge-bg {
          fill: none;
          stroke: rgba(255, 255, 255, 0.08);
          stroke-width: 12;
        }

        .gauge-fill {
          fill: none;
          stroke-width: 12;
          stroke-linecap: round;
          transition: stroke-dashoffset 0.6s ease;
        }

        .gauge-fill--green {
          stroke: var(--accent-green);
        }

        .gauge-fill--amber {
          stroke: var(--status-amber);
        }

        .gauge-fill--red {
          stroke: var(--status-red);
        }

        .gauge-value-text {
          font-size: 2.75rem;
          font-weight: 750;
          fill: var(--text-primary);
          dominant-baseline: middle;
          text-anchor: middle;
          transform: rotate(90deg);
          transform-origin: 90px 90px;
          font-variant-numeric: tabular-nums;
        }

        .status-pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 5px 14px;
          border-radius: var(--radius-sm);
          font-size: 0.84rem;
          font-weight: 600;
          letter-spacing: 0.01em;
          margin-top: var(--space-3);
        }

        .status-pill--danger {
          background: var(--status-red-bg);
          border: 1px solid var(--status-red-border);
          color: var(--status-red);
        }

        .status-pill--warning {
          background: var(--status-amber-bg);
          border: 1px solid var(--status-amber-border);
          color: var(--status-amber);
        }

        .status-pill--success {
          background: var(--status-green-bg);
          border: 1px solid var(--status-green-border);
          color: var(--status-green);
        }

        /* Stat Grid & Metric Cards */
        .stat-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
          gap: var(--space-2);
          margin-top: var(--space-3);
        }

        .stat-card {
          background: var(--bg-subtle);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-sm);
          padding: var(--space-3);
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .stat-card-label {
          font-size: 0.74rem;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.04em;
          font-weight: 600;
          display: flex;
          align-items: center;
          gap: 5px;
        }

        .stat-card-value {
          font-size: 1.35rem;
          font-weight: 700;
          color: var(--text-primary);
          font-variant-numeric: tabular-nums;
        }

        /* Contributor Bars */
        .contrib-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
          margin-top: var(--space-2);
        }

        .contrib-row {
          background: var(--bg-subtle);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-sm);
          padding: 10px 14px;
        }

        .contrib-row-top {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
        }

        .contrib-feature-name {
          font-size: 0.86rem;
          font-weight: 600;
          color: var(--text-primary);
        }

        .contrib-feature-value {
          font-size: 0.8rem;
          color: var(--text-secondary);
          font-variant-numeric: tabular-nums;
        }

        .contrib-track-wrap {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        .contrib-track {
          flex: 1;
          height: 6px;
          background: rgba(255, 255, 255, 0.08);
          border-radius: 3px;
          overflow: hidden;
        }

        .contrib-fill {
          height: 100%;
          border-radius: 3px;
          background: var(--accent-green);
        }

        .contrib-pct {
          font-size: 0.78rem;
          font-weight: 600;
          color: var(--text-primary);
          min-width: 44px;
          text-align: right;
          font-variant-numeric: tabular-nums;
        }

        /* Clean SaaS Zebra Table */
        .saas-table-wrap {
          overflow-x: auto;
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-sm);
          margin-top: var(--space-2);
        }

        .saas-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.85rem;
          text-align: left;
        }

        .saas-table th {
          background: var(--bg-subtle);
          padding: 10px 14px;
          color: var(--text-secondary);
          font-weight: 600;
          font-size: 0.76rem;
          text-transform: uppercase;
          letter-spacing: 0.03em;
          border-bottom: 1px solid var(--border-subtle);
        }

        .saas-table td {
          padding: 10px 14px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.04);
          color: var(--text-primary);
        }

        .saas-table tbody tr:nth-child(even) {
          background: rgba(255, 255, 255, 0.015);
        }

        .saas-table tbody tr:hover {
          background: rgba(16, 185, 129, 0.04);
        }

        /* Clinical Guidance Boxes */
        .guidance-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-3);
          margin-top: var(--space-3);
        }

        .guidance-box {
          background: var(--bg-subtle);
          border-left: 3px solid var(--accent-green);
          border-top: 1px solid var(--border-subtle);
          border-right: 1px solid var(--border-subtle);
          border-bottom: 1px solid var(--border-subtle);
          border-radius: var(--radius-sm);
          padding: var(--space-3);
        }

        .guidance-box--alert {
          border-left-color: var(--status-amber);
        }

        .guidance-box-title {
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--text-primary);
          margin: 0 0 6px 0;
        }

        .guidance-box-text {
          font-size: 0.82rem;
          color: var(--text-secondary);
          line-height: 1.45;
          margin: 0;
        }

        /* Empty State */
        .saas-empty {
          text-align: center;
          padding: var(--space-5) var(--space-4);
          background: var(--bg-card);
          border: 1px dashed var(--border-strong);
          border-radius: var(--radius-md);
        }

        .saas-empty-title {
          font-size: 1.05rem;
          font-weight: 600;
          color: var(--text-primary);
          margin: var(--space-2) 0 4px 0;
        }

        .saas-empty-desc {
          font-size: 0.85rem;
          color: var(--text-secondary);
          margin: 0;
          max-width: 440px;
          margin-left: auto;
          margin-right: auto;
          line-height: 1.45;
        }

        /* Clinical Disclaimer */
        .clinical-disclaimer {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: 10px 14px;
          background: rgba(255, 255, 255, 0.02);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-sm);
          font-size: 0.78rem;
          color: var(--text-muted);
          margin-top: var(--space-4);
        }

        /* Mobile Adjustments */
        @media (max-width: 768px) {
          .main .block-container {
            padding: var(--space-2);
          }
          .saas-topbar {
            flex-direction: column;
            align-items: flex-start;
            gap: var(--space-2);
          }
          .guidance-grid {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# SVG Graphic Assets (Medical & Metrics)
# =====================================================================


def _svg_shield() -> str:
    return (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
        '<line x1="12" y1="8" x2="12" y2="14"/>'
        '<line x1="9" y1="11" x2="15" y2="11"/>'
        '</svg>'
    )


def _svg_capsule() -> str:
    return (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/>'
        '<path d="m8.5 8.5 7 7"/>'
        '</svg>'
    )


def _svg_alert() -> str:
    return (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/>'
        '</svg>'
    )


def _svg_pulse() -> str:
    return (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
        '</svg>'
    )


def _svg_network() -> str:
    return (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<rect x="2" y="2" width="6" height="6" rx="1"/>'
        '<rect x="16" y="2" width="6" height="6" rx="1"/>'
        '<rect x="9" y="16" width="6" height="6" rx="1"/>'
        '<path d="M5 8v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8"/>'
        '<line x1="12" y1="13" x2="12" y2="16"/>'
        '</svg>'
    )


# =====================================================================
# Top Bar & Header Component
# =====================================================================


def _render_topbar() -> None:
    """Render branded top navigation bar."""
    st.markdown(
        f"""
        <header class="saas-topbar">
          <div class="saas-brand">
            <div class="saas-logo-wrap" aria-hidden="true">
              {_svg_shield()}
            </div>
            <div>
              <h1 class="saas-brand-title">TekaRx</h1>
              <p class="saas-brand-subtitle">
                Clinical Decision Support : Medication Safety Review
              </p>
            </div>
          </div>
          <div class="saas-badges">
            <span class="saas-badge">
              <span class="saas-badge-dot" aria-hidden="true"></span>
              IMRAD Serial Frozen Model
            </span>
            <span class="saas-badge">
              Research Prototype
            </span>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# Main Application Flow
# =====================================================================


def main() -> None:
    """Primary application controller."""
    st.set_page_config(
        page_title="TekaRx Medication Priority Review",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Initialize Session State
    if "input_age" not in st.session_state:
        st.session_state["input_age"] = 65
    if "input_sex" not in st.session_state:
        st.session_state["input_sex"] = "Female"
    if "input_meds" not in st.session_state:
        st.session_state["input_meds"] = "Aspirin\nMetformin\nAtorvastatin"
    if "prefill_age" not in st.session_state:
        st.session_state["prefill_age"] = 65
    if "prefill_sex" not in st.session_state:
        st.session_state["prefill_sex"] = "Female"
    if "prefill_meds" not in st.session_state:
        st.session_state["prefill_meds"] = "Aspirin\nMetformin\nAtorvastatin"
    if "selected_drugs" not in st.session_state:
        st.session_state["selected_drugs"] = ["ASPIRIN", "METFORMIN", "ATORVASTATIN"]
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None
    if "auto_submit" not in st.session_state:
        st.session_state["auto_submit"] = False

    _render_css()
    _render_topbar()

    # Artifact Loading
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error(
            f"Unable to load frozen TekaRx artifacts. "
            f"Build them via:\n{BUILD_COMMANDS}\n\nError details: {exc}"
        )
        return

    bundle = artifacts["bundle"]
    dictionary = artifacts["dictionary"]
    threshold = _threshold(bundle)

    # Top Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["New Check", "Results & Contributors", "Mapping Coverage", "About"]
    )

    with tab1:
        # Preset Quick-Loads
        st.markdown(
            '<p style="font-size: 0.8rem; color: var(--text-secondary); '
            'margin-bottom: 6px; text-transform: uppercase; '
            'letter-spacing: 0.03em; font-weight: 600;">'
            "Sample Regimens & Synthesized Scenarios</p>",
            unsafe_allow_html=True,
        )
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button(
                "Load High-Priority Polypharmacy Example",
                key="btn_high_example",
                use_container_width=True,
            ):
                high_meds = "Warfarin\nIbuprofen\nTramadol\nMetoprolol\nOmeprazole"
                st.session_state["input_age"] = 74
                st.session_state["input_sex"] = "Female"
                st.session_state["input_meds"] = high_meds
                st.session_state["prefill_age"] = 74
                st.session_state["prefill_sex"] = "Female"
                st.session_state["prefill_meds"] = high_meds
                st.session_state["selected_drugs"] = [
                    "WARFARIN",
                    "IBUPROFEN",
                    "TRAMADOL",
                    "METOPROLOL",
                    "OMEPRAZOLE",
                ]
                st.session_state["auto_submit"] = True
                st.rerun()

        with btn_c2:
            if st.button(
                "Load Maintenance Monotherapy Example",
                key="btn_low_example",
                use_container_width=True,
            ):
                low_meds = "Metformin\nAtorvastatin\nLevothyroxine"
                st.session_state["input_age"] = 52
                st.session_state["input_sex"] = "Male"
                st.session_state["input_meds"] = low_meds
                st.session_state["prefill_age"] = 52
                st.session_state["prefill_sex"] = "Male"
                st.session_state["prefill_meds"] = low_meds
                st.session_state["selected_drugs"] = [
                    "METFORMIN",
                    "ATORVASTATIN",
                    "LEVOTHYROXINE",
                ]
                st.session_state["auto_submit"] = True
                st.rerun()

        with btn_c3:
            if st.button(
                "Generate Random Record",
                key="btn_generate_record",
                use_container_width=True,
            ):
                age_gen, sex_gen, meds_gen, _ = _generate_record(
                    bundle, dictionary
                )
                valid_sex = (
                    sex_gen if sex_gen in ("Female", "Male", "Unknown") else "Unknown"
                )
                st.session_state["input_age"] = int(age_gen)
                st.session_state["input_sex"] = valid_sex
                st.session_state["input_meds"] = meds_gen
                st.session_state["prefill_age"] = int(age_gen)
                st.session_state["prefill_sex"] = valid_sex
                st.session_state["prefill_meds"] = meds_gen
                st.session_state["selected_drugs"] = _parse_medications(
                    meds_gen
                )
                st.session_state["auto_submit"] = True
                st.rerun()

        st.write("")

        col_left, col_right = st.columns([1.1, 1], gap="large")

        with col_left:
            st.markdown(
                '<div class="saas-card-header" style="margin-bottom: 8px;">'
                f'<h2 class="saas-card-title">{_svg_capsule()} Patient Regimen Entry</h2>'
                '</div>',
                unsafe_allow_html=True,
            )

            c_age, c_sex = st.columns(2)
            with c_age:
                age_val = st.number_input(
                    "Patient Age (years)",
                    min_value=0,
                    max_value=120,
                    step=1,
                    key="input_age",
                )
            with c_sex:
                sex_options = ("Female", "Male", "Unknown")
                sex_val = st.selectbox(
                    "Biological Sex",
                    sex_options,
                    key="input_sex",
                )

            # Autocomplete & Quick-Add
            st.markdown(
                '<p style="font-size: 0.84rem; color: var(--text-secondary); '
                'margin: 12px 0 4px 0; font-weight: 500;">'
                "Search & Quick-Add Medication</p>",
                unsafe_allow_html=True,
            )
            search_query = st.text_input(
                "Medication search",
                key="drug_search_box",
                placeholder="Type drug prefix (e.g. Warfarin, Aspirin)...",
                label_visibility="collapsed",
            )

            if search_query:
                suggestions = _get_drug_suggestions(
                    search_query, dictionary, limit=6
                )
                if suggestions:
                    st.caption("Click a suggested medication to add it immediately:")
                    sug_cols = st.columns(min(len(suggestions), 3))
                    for i, drug_sug in enumerate(suggestions):
                        target_col = sug_cols[i % len(sug_cols)]
                        with target_col:
                            if st.button(
                                f"+ {drug_sug}",
                                key=f"sug_add_{i}_{drug_sug}",
                                use_container_width=True,
                            ):
                                current_raw = st.session_state.get("input_meds", "")
                                existing_list = _parse_medications(current_raw)
                                if drug_sug not in existing_list:
                                    existing_list.append(drug_sug)
                                    new_meds_text = "\n".join(existing_list)
                                    st.session_state["input_meds"] = new_meds_text
                                    st.session_state["prefill_meds"] = new_meds_text
                                    if drug_sug not in st.session_state.get("selected_drugs", []):
                                        st.session_state["selected_drugs"].append(drug_sug)
                                st.rerun()

            # Active Regimen Text Area
            meds_input = st.text_area(
                "Active Medications (one per line or comma-separated)",
                height=140,
                placeholder="Aspirin\nMetformin\nAtorvastatin",
                key="input_meds",
            )

            # Action Button
            run_clicked = st.button(
                "Run TekaRx Priority Score",
                type="primary",
                use_container_width=True,
                key="btn_run_score",
            )

        # Regimen execution trigger
        should_run = run_clicked or st.session_state.get("auto_submit", False)
        if should_run:
            st.session_state["auto_submit"] = False
            active_meds_text = st.session_state.get("input_meds", meds_input)
            parsed_meds = _parse_medications(active_meds_text)
            if not parsed_meds:
                st.warning("Please enter at least one active medication.")
            else:
                with st.spinner("Analyzing regimen across IMRAD models..."):
                    matched_df, match_info = _match_medications(
                        parsed_meds, dictionary
                    )
                    features_df, summary_stats = _build_features(
                        float(st.session_state["input_age"]),
                        str(st.session_state["input_sex"]),
                        matched_df,
                        bundle,
                    )
                    score_prob = _predict(features_df, artifacts)
                    contrib_df = _top_contributors(bundle, features_df)

                    st.session_state["last_prediction"] = {
                        "probability": score_prob,
                        "threshold": threshold,
                        "matched": matched_df,
                        "match_details": match_info,
                        "frame": features_df,
                        "summary": summary_stats,
                        "contributors": contrib_df,
                        "age": float(st.session_state["input_age"]),
                        "sex": str(st.session_state["input_sex"]),
                        "parsed_meds": parsed_meds,
                    }

        with col_right:
            pred = st.session_state["last_prediction"]
            if not pred:
                st.markdown(
                    f"""
                    <div class="saas-empty">
                      <div style="color: var(--text-muted); margin-bottom: 8px;">
                        {_svg_pulse()}
                      </div>
                      <h3 class="saas-empty-title">Ready for Clinical Evaluation</h3>
                      <p class="saas-empty-desc">
                        Enter patient demographics and active medications on the left,
                        or load one of the test scenarios to inspect model priority scores.
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                prob = pred["probability"]
                is_alert = prob >= pred["threshold"]
                status_cls = (
                    "status-pill--danger" if is_alert else "status-pill--success"
                )
                gauge_cls = (
                    "gauge-fill--amber" if is_alert else "gauge-fill--green"
                )
                status_text = (
                    "Review Priority Signal"
                    if is_alert
                    else "No Priority Signal"
                )
                dashoffset = max(
                    0, min(440, int(440 * (1.0 - prob)))
                )
                m_cnt = pred["summary"]["Matched medications"]
                h_ror = pred["summary"]["Highest ROR"]
                box_w = pred["summary"]["Boxed warning"]

                st.markdown(
                    f"""
                    <div class="saas-card saas-card--elevated">
                      <div class="saas-card-header">
                        <h3 class="saas-card-title">{_svg_pulse()} Priority Assessment</h3>
                        <span style="font-size: 0.78rem; color: var(--text-secondary);">
                          Threshold: {pred['threshold']:.0%}
                        </span>
                      </div>
                      <div class="score-hero">
                        <svg viewBox="0 0 180 180" class="gauge-svg">
                          <circle cx="90" cy="90" r="70" class="gauge-bg" />
                          <circle cx="90" cy="90" r="70"
                            class="gauge-fill {gauge_cls}"
                            stroke-dasharray="440"
                            stroke-dashoffset="{dashoffset}" />
                          <text x="90" y="90" class="gauge-value-text">{prob:.0%}</text>
                        </svg>
                        <div class="status-pill {status_cls}">
                          {status_text}
                        </div>
                      </div>
                      <div class="stat-grid">
                        <div class="stat-card">
                          <span class="stat-card-label">{_svg_capsule()} Matched</span>
                          <span class="stat-card-value">{m_cnt}</span>
                        </div>
                        <div class="stat-card">
                          <span class="stat-card-label">{_svg_alert()} Max ROR</span>
                          <span class="stat-card-value">{h_ror:.2f}</span>
                        </div>
                        <div class="stat-card">
                          <span class="stat-card-label">{_svg_network()} Boxed</span>
                          <span class="stat-card-value">{box_w}</span>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------
    # Tab 2: Results & Contributors
    # -----------------------------------------------------------------
    with tab2:
        pred = st.session_state.get("last_prediction")
        if not pred:
            st.markdown(
                f"""
                <div class="saas-empty">
                  <div style="color: var(--text-muted); margin-bottom: 8px;">
                    {_svg_pulse()}
                  </div>
                  <h3 class="saas-empty-title">No Evaluation Results</h3>
                  <p class="saas-empty-desc">
                    Execute a regimen assessment in the "New Check" tab to view
                    top contributing features, risk signals, and clinical guidance.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            prob = pred["probability"]
            is_alert = prob >= pred["threshold"]
            guidance_meaning = (
                "Model priority signal detected under the frozen Random Forest weights. "
                "Medication combination exhibits elevated statistical reporting odds ratios "
                "or complex multi-drug interaction patterns."
                if is_alert
                else "No high-priority safety signal detected under the frozen IMRAD model. "
                "Regimen parameters fall within baseline expected tolerance."
            )
            guidance_action = (
                "Prioritize formal clinical pharmacotherapy review. Inspect highlighted "
                "drug pairs, cumulative toxicities, and potential therapeutic duplicates."
                if is_alert
                else "Maintain routine clinical pharmacotherapy monitoring. Re-evaluate if "
                "dosages, active compounds, or patient renal/hepatic markers change."
            )

            col_res_l, col_res_r = st.columns([1, 1.2], gap="large")

            with col_res_l:
                score_color = (
                    "var(--status-amber)" if is_alert else "var(--accent-green)"
                )
                pill_cls = (
                    "status-pill--danger" if is_alert else "status-pill--success"
                )
                pill_lbl = (
                    "Review Priority Signal" if is_alert else "No Priority Signal"
                )
                guidance_alert_cls = "guidance-box--alert" if is_alert else ""

                st.markdown(
                    f"""
                    <div class="saas-card saas-card--elevated">
                      <div class="saas-card-header">
                        <h3 class="saas-card-title">{_svg_pulse()} Overall Risk Signal</h3>
                        <span style="font-size: 0.8rem; color: var(--text-secondary);">
                          Operating Threshold: {pred['threshold']:.0%}
                        </span>
                      </div>
                      <div class="score-hero" style="padding: 10px 0;">
                        <span style="font-size: 3.5rem; font-weight: 800; color: {score_color};">
                          {prob:.0%}
                        </span>
                        <div class="status-pill {pill_cls}">
                          {pill_lbl}
                        </div>
                      </div>
                      <div class="guidance-grid" style="grid-template-columns: 1fr;">
                        <div class="guidance-box {guidance_alert_cls}">
                          <h4 class="guidance-box-title">Signal Interpretation</h4>
                          <p class="guidance-box-text">{guidance_meaning}</p>
                        </div>
                        <div class="guidance-box">
                          <h4 class="guidance-box-title">Recommended Clinical Step</h4>
                          <p class="guidance-box-text">{guidance_action}</p>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_res_r:
                contrib_rows = pred["contributors"].to_dict("records")
                contrib_html_items = []
                for item in contrib_rows:
                    pct_str = item["Contribution"]
                    pct_num = (
                        float(pct_str.rstrip("%"))
                        if pct_str != "N/A"
                        else 0.0
                    )
                    contrib_html_items.append(
                        f"""
                        <div class="contrib-row">
                          <div class="contrib-row-top">
                            <span class="contrib-feature-name">{item['Feature']}</span>
                            <span class="contrib-feature-value">{item['Value']}</span>
                          </div>
                          <div class="contrib-track-wrap">
                            <div class="contrib-track">
                              <div class="contrib-fill" style="width: {pct_num}%;"></div>
                            </div>
                            <span class="contrib-pct">{pct_str}</span>
                          </div>
                        </div>
                        """
                    )

                contrib_inner = "\n".join(item.strip() for item in contrib_html_items)
                st.markdown(
                    f"""
                    <div class="saas-card">
                      <div class="saas-card-header">
                        <h3 class="saas-card-title">{_svg_network()} Top Contributing Factors</h3>
                        <span style="font-size: 0.78rem; color: var(--text-secondary);">
                          Relative Importance
                        </span>
                      </div>
                      <div class="contrib-list">
                        {contrib_inner}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Clinical Disclaimer
            st.markdown(
                f"""
                <div class="clinical-disclaimer">
                  {_svg_shield()}
                  <span>
                    <strong>Clinical Reminder:</strong> {DISCLAIMER} Always exercise judgment.
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------
    # Tab 3: Mapping Coverage
    # -----------------------------------------------------------------
    with tab3:
        pred = st.session_state.get("last_prediction")
        if not pred:
            st.markdown(
                f"""
                <div class="saas-empty">
                  <div style="color: var(--text-muted); margin-bottom: 8px;">
                    {_svg_capsule()}
                  </div>
                  <h3 class="saas-empty-title">No Medication Mapping Available</h3>
                  <p class="saas-empty-desc">
                    Enter medications and execute an evaluation in "New Check"
                    to review RapidFuzz coverage against the FAERS standard vocabulary.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            match_details = pred["match_details"]
            matched_count = sum(
                1 for m in match_details if m["status"] == "matched"
            )
            ambig_count = sum(
                1 for m in match_details if m["status"] == "ambiguous"
            )
            unmatched_count = sum(
                1 for m in match_details if m["status"] == "unmatched"
            )

            p_style = "margin: 0; padding: 4px 12px; font-size: 0.78rem; font-weight: 600;"

            table_rows_html = []
            for m in match_details:
                st_cls = (
                    "status-pill--success"
                    if m["status"] == "matched"
                    else (
                        "status-pill--warning"
                        if m["status"] == "ambiguous"
                        else "status-pill--danger"
                    )
                )
                label_text = m["status"].capitalize()
                match_val = (
                    f"<code>{html.escape(str(m['match']))}</code>"
                    if m["match"]
                    else "<span style='color: var(--text-muted); font-style: italic;'>Unmapped</span>"
                )
                used_badge = (
                    '<span style="color: var(--accent-green); font-weight: 700;">✓ Included</span>'
                    if m["used"]
                    else '<span style="color: var(--text-muted); font-weight: 500;">✕ Excluded</span>'
                )
                suggestions_escaped = html.escape(str(m["suggestions"]))
                pill_style = "margin: 0; padding: 3px 10px; font-size: 0.74rem; font-weight: 600; display: inline-block;"

                score_pct = int(round(m["score"]))
                score_bar_color = (
                    "var(--accent-green)"
                    if m["status"] == "matched"
                    else (
                        "var(--status-amber)"
                        if m["status"] == "ambiguous"
                        else "var(--status-red)"
                    )
                )

                score_cell_html = f"""
                <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
                  <div style="width: 54px; height: 5px; background: rgba(255, 255, 255, 0.08); border-radius: 3px; overflow: hidden;">
                    <div style="width: {score_pct}%; height: 100%; background: {score_bar_color}; border-radius: 3px;"></div>
                  </div>
                  <span style="font-weight: 600; min-width: 34px; text-align: right; font-variant-numeric: tabular-nums;">{score_pct}%</span>
                </div>
                """

                table_rows_html.append(
                    f"""
                    <tr>
                      <td style="font-weight: 600; color: var(--text-primary); font-size: 0.88rem;">{html.escape(str(m['input']))}</td>
                      <td>
                        <span class="status-pill {st_cls}" style="{pill_style}">
                          {label_text}
                        </span>
                      </td>
                      <td>{match_val}</td>
                      <td>{score_cell_html}</td>
                      <td style="text-align: center;">{used_badge}</td>
                      <td style="font-size: 0.8rem; color: var(--text-secondary); max-width: 280px; word-break: break-word;">
                        {suggestions_escaped}
                      </td>
                    </tr>
                    """
                )

            rendered_rows = "\n".join(item.strip() for item in table_rows_html)
            table_card_html = f"""
            <div class="saas-card">
              <div class="saas-card-header" style="flex-wrap: wrap; gap: 12px;">
                <div>
                  <h3 class="saas-card-title">{_svg_capsule()} FAERS Vocabulary Mapping Coverage</h3>
                  <p style="font-size: 0.82rem; color: var(--text-secondary); margin: 4px 0 0 0;">
                    RapidFuzz alignment against FAERS standard lexicon (Matched threshold: {MATCHED_CUTOFF:.0f}%, Ambiguous: {AMBIGUOUS_CUTOFF:.0f}%)
                  </p>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                  <span class="status-pill status-pill--success" style="{p_style}">
                    {matched_count} Matched
                  </span>
                  <span class="status-pill status-pill--warning" style="{p_style}">
                    {ambig_count} Ambiguous
                  </span>
                  <span class="status-pill status-pill--danger" style="{p_style}">
                    {unmatched_count} Unmatched
                  </span>
                </div>
              </div>
              <div class="saas-table-wrap">
                <table class="saas-table">
                  <thead>
                    <tr>
                      <th>Input Query</th>
                      <th>Mapping Status</th>
                      <th>Standard Lexicon Match</th>
                      <th style="text-align: right;">Confidence</th>
                      <th style="text-align: center;">Model Vector</th>
                      <th>Lexicon Match Candidates</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rendered_rows}
                  </tbody>
                </table>
              </div>
            </div>
            """
            st.markdown(table_card_html, unsafe_allow_html=True)

            if unmatched_count > 0 or ambig_count > 0:
                st.info(
                    f"{unmatched_count + ambig_count} medication(s) had confidence "
                    f"below {MATCHED_CUTOFF:.0f}%. "
                    "Verify spelling or pick the exact compound from the candidate list for optimal predictive power."
                )

    # -----------------------------------------------------------------
    # Tab 4: About & Methodology
    # -----------------------------------------------------------------
    with tab4:
        st.markdown(
            f"""
            <div class="saas-card">
              <h3 class="saas-card-title">{_svg_shield()} Model Architecture & Governance</h3>
              <p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.5;">
                TekaRx evaluates polypharmacy regimens using a frozen ensemble classifier
                trained on the FAERS integrated pharmacovigilance cohort. The model quantifies
                reporting odds ratio (ROR) associations, multi-drug interactions, and anatomical
                therapeutic chemical (ATC) classifications.
              </p>
              <div class="stat-grid" style="margin: 16px 0;">
                <div class="stat-card">
                  <span class="stat-card-label">Frozen Pipeline</span>
                  <span class="stat-card-value">Random Forest</span>
                </div>
                <div class="stat-card">
                  <span class="stat-card-label">Feature Vector</span>
                  <span class="stat-card-value">{len(bundle['feature_cols'])} Dimensions</span>
                </div>
                <div class="stat-card">
                  <span class="stat-card-label">Operating Threshold</span>
                  <span class="stat-card-value">{threshold:.0%}</span>
                </div>
                <div class="stat-card">
                  <span class="stat-card-label">Dictionary Scope</span>
                  <span class="stat-card-value">{len(dictionary):,} Compounds</span>
                </div>
              </div>
            </div>

            <div class="guidance-grid">
              <div class="saas-card">
                <h4 class="saas-card-title">{_svg_network()} Feature Extraction Pipeline</h4>
                <p style="font-size: 0.84rem; color: var(--text-secondary); line-height: 1.45;">
                  Active medications are parsed and matched against normalized dictionary compounds.
                  Extracted features include ROR statistics, cumulative drug counts, high-risk ATC
                  classes (nervous, cardiovascular, blood, musculoskeletal), and overlaps.
                </p>
              </div>
              <div class="saas-card">
                <h4 class="saas-card-title">{_svg_alert()} Limitations & Intended Use</h4>
                <p style="font-size: 0.84rem; color: var(--text-secondary); line-height: 1.45;">
                  The model reflects reporting signals present in spontaneous databases.
                  It does not account for dosages, patient renal function, or direct causation.
                  Results serve strictly as a research decision-support aid.
                </p>
              </div>
            </div>

            <div class="clinical-disclaimer">
              {_svg_shield()}
              <span>{DISCLAIMER} Not for independent diagnostic decision-making.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
