"""Modern SaaS-grade Streamlit dashboard for frozen TekaRx IMRAD models."""

from __future__ import annotations

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

DISCLAIMER = "Research decision-support, not medical advice."
MODEL_KEY = "random_forest"
DEFAULT_THRESHOLD = 0.50
MATCHED_CUTOFF = 90.0
AMBIGUOUS_CUTOFF = 75.0

BUILD_COMMANDS = (
    "python -m pip install -e .[imrad,notebook]\n"
    "python build_imrad_artifacts.py"
)


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
        raise ValueError(f"Drug dictionary is missing columns: {', '.join(missing)}")


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

    # Use partial_ratio for prefix matching and autocomplete
    candidates = process.extract(
        query_upper, choices, scorer=fuzz.partial_ratio, limit=limit
    )
    # Filter by score >= 70 for quality suggestions
    return [candidate[0] for candidate in candidates if candidate[1] >= 70]


def _match_medications(
    queries: list[str], dictionary: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Match medication queries against dictionary."""
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
            rows.extend(dictionary.loc[dictionary["faers_raw"] == match].to_dict(
                "records"
            ))
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


def _feature_defaults(bundle: dict[str, Any]) -> dict[str, float]:
    values = bundle["imputer"].statistics_
    return dict(zip(bundle["feature_cols"], values, strict=True))


def _build_features(
    age: float,
    sex: str,
    matched: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build feature frame from matched drugs and demographics."""
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
            pd.to_numeric(
                matched["has_boxed_warning"], errors="coerce"
            ).fillna(0).max()
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
                sum(code[:4] in {"N02A", "B01A", "M01A", "A10A"}
                    for code in unique_atc)
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
    """Run prediction."""
    bundle = artifacts["bundle"]
    aligned = frame.reindex(columns=bundle["feature_cols"])
    transformed = bundle["scaler"].transform(bundle["imputer"].transform(aligned))
    return float(bundle[MODEL_KEY].predict_proba(transformed)[0, 1])


def _threshold(bundle: dict[str, Any]) -> float:
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


def _top_contributors(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Get top 5 contributing features."""
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
    """Get human-readable feature name."""
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
    """Format feature value for display."""
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
    }:
        return f"{int(round(val))}"
    if name == "age_imputed_years":
        return f"{val:.0f} yr"
    if "ror" in name:
        return f"{val:.2f}"
    return f"{val:.2f}"


def _render_css() -> None:
    """Render modern SaaS design CSS."""
    st.markdown(
        """
        <style>
        :root {
          --ink: #16221c;
          --ink-light: #485750;
          --border: #e3ece6;
          --bg-panel: #f9faf8;
          --green-primary: #1e5b3a;
          --green-light: #eaf3ed;
          --green-border: #c3ddcc;
          --amber-primary: #8a580a;
          --amber-light: #fef6e7;
          --red-primary: #9b2828;
          --red-light: #fbeeed;
          --white: #ffffff;
        }

        * { box-sizing: border-box; }

        .stApp {
          background: var(--white);
          color: var(--ink);
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            sans-serif;
        }

        .block-container {
          max-width: 1280px;
          padding: 2rem;
        }

        /* Header */
        .app-header {
          margin-bottom: 2.5rem;
          border-bottom: 1px solid var(--border);
          padding-bottom: 1.5rem;
        }

        .app-header h1 {
          font-size: clamp(1.8rem, 4vw, 2.4rem);
          font-weight: 700;
          margin: 0 0 0.25rem;
          letter-spacing: -0.02em;
          color: var(--ink);
        }

        .app-header p {
          font-size: 0.95rem;
          color: var(--ink-light);
          margin: 0;
          line-height: 1.4;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
          border-bottom: 1px solid var(--border);
          gap: 0.5rem;
        }

        .stTabs [aria-selected="true"] {
          border-bottom: 3px solid var(--green-primary) !important;
          color: var(--green-primary) !important;
        }

        .stTabs [aria-selected="false"] {
          color: var(--ink-light);
        }

        /* Cards */
        .card {
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 1.5rem;
          background: var(--white);
          box-shadow: 0 1px 3px rgba(22, 34, 28, 0.08);
        }

        .card-sm {
          padding: 1rem;
        }

        .card h3 {
          font-size: 0.9rem;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: var(--ink-light);
          margin: 0 0 0.5rem;
          font-weight: 600;
        }

        /* Input styling */
        .drug-input-container {
          margin-bottom: 1.5rem;
        }

        .drug-input-label {
          display: block;
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--ink);
          margin-bottom: 0.5rem;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }

        .drug-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-bottom: 0.75rem;
          min-height: 1.5rem;
        }

        .chip {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          background: var(--green-light);
          color: var(--green-primary);
          border: 1px solid var(--green-border);
          border-radius: 6px;
          padding: 0.35rem 0.75rem;
          font-size: 0.85rem;
          font-weight: 500;
        }

        .chip-remove {
          cursor: pointer;
          font-weight: bold;
          opacity: 0.7;
          transition: opacity 0.2s;
        }

        .chip-remove:hover {
          opacity: 1;
        }

        /* Score display */
        .score-section {
          background: var(--bg-panel);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 2rem;
          text-align: center;
          margin-bottom: 2rem;
        }

        .score-value {
          font-size: clamp(3rem, 8vw, 5rem);
          font-weight: 750;
          color: var(--green-primary);
          line-height: 1;
          margin: 0;
        }

        .score-value.high {
          color: var(--amber-primary);
        }

        .priority-badge {
          display: inline-block;
          margin-top: 0.75rem;
          padding: 0.4rem 1rem;
          background: var(--green-light);
          color: var(--green-primary);
          border: 1px solid var(--green-border);
          border-radius: 6px;
          font-size: 0.9rem;
          font-weight: 600;
        }

        .priority-badge.high {
          background: var(--amber-light);
          color: var(--amber-primary);
          border-color: #f2dcab;
        }

        .threshold-note {
          font-size: 0.85rem;
          color: var(--ink-light);
          margin-top: 0.75rem;
        }

        /* Guidance cards */
        .guidance-card {
          border-left: 4px solid var(--green-primary);
          background: var(--green-light);
          padding: 1rem;
          border-radius: 4px;
          margin-bottom: 1rem;
        }

        .guidance-card h4 {
          margin: 0 0 0.5rem;
          font-size: 0.9rem;
          color: var(--green-primary);
          font-weight: 600;
        }

        .guidance-card p {
          margin: 0;
          font-size: 0.9rem;
          color: var(--ink);
          line-height: 1.5;
        }

        /* Tables */
        .stDataFrame {
          font-size: 0.85rem !important;
        }

        /* Status grid */
        .status-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1rem;
          margin: 1.5rem 0;
        }

        .status-item {
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 1rem;
          background: var(--bg-panel);
        }

        .status-label {
          font-size: 0.75rem;
          color: var(--ink-light);
          text-transform: uppercase;
          letter-spacing: 0.03em;
          margin-bottom: 0.4rem;
          font-weight: 600;
        }

        .status-value {
          font-size: 1.25rem;
          font-weight: 700;
          color: var(--ink);
        }

        /* Buttons */
        .stButton > button {
          border-radius: 6px;
          border: 1px solid var(--green-primary);
          background: var(--green-primary);
          color: var(--white);
          font-weight: 600;
          font-size: 0.95rem;
          padding: 0.75rem 1.5rem !important;
          transition: all 0.2s;
        }

        .stButton > button:hover {
          background: #164a2f;
          border-color: #164a2f;
        }

        /* Disclaimer */
        .disclaimer {
          border-left: 3px solid var(--ink-light);
          padding-left: 1rem;
          font-size: 0.8rem;
          color: var(--ink-light);
          margin-top: 1rem;
        }

        /* Empty state */
        .empty-state {
          text-align: center;
          padding: 3rem 2rem;
          border: 2px dashed var(--border);
          border-radius: 8px;
          background: var(--bg-panel);
        }

        .empty-state h3 {
          font-size: 1.2rem;
          color: var(--ink);
          margin: 0 0 0.5rem;
        }

        .empty-state p {
          color: var(--ink-light);
          font-size: 0.9rem;
          margin: 0;
          line-height: 1.5;
        }

        /* Mobile responsive */
        @media (max-width: 768px) {
          .block-container {
            padding: 1rem;
          }

          .app-header {
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
          }

          .app-header h1 {
            font-size: 1.5rem;
          }

          .score-section {
            padding: 1.5rem;
          }

          .score-value {
            font-size: 2.4rem;
          }

          .status-grid {
            grid-template-columns: 1fr;
          }

          .card {
            padding: 1rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    """Render page header."""
    st.markdown(
        """
        <div class="app-header">
          <h1>TekaRx Medication Priority Check</h1>
          <p>Enter patient age, sex, and medications to evaluate the Model Priority
          Score against the frozen IMRAD seriousness model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _drug_autocomplete(dictionary: pd.DataFrame) -> str:
    """Render drug autocomplete input. Returns selected drug name."""
    st.markdown(
        '<div class="drug-input-label">Search & Add Medications</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1], gap="small")
    with col1:
        search_term = st.text_input(
            "Type drug name",
            key="drug_search",
            placeholder="e.g., Warfarin, Aspirin...",
            label_visibility="collapsed",
        )

    if not search_term:
        return ""

    suggestions = _get_drug_suggestions(search_term, dictionary)
    if not suggestions:
        st.caption("No matches found")
        return ""

    with col2:
        st.write("")  # Spacer
        st.write("")

    # Show suggestions as buttons
    for i, drug in enumerate(suggestions[:8]):
        if st.button(drug, key=f"drug_{i}"):
            st.session_state["selected_drugs"].append(drug)
            st.rerun()

    return ""


def _render_selected_chips() -> None:
    """Render selected drug chips."""
    if not st.session_state.get("selected_drugs"):
        return

    st.markdown('<div class="drug-chips">', unsafe_allow_html=True)

    drugs = st.session_state.get("selected_drugs", [])
    for drug in drugs:
        col1, col2 = st.columns([1, 0.1], gap="small")
        with col1:
            st.write(f"🔹 {drug}")
        with col2:
            if st.button("✕", key=f"remove_{drug}"):
                st.session_state["selected_drugs"].remove(drug)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    """Main app entry point."""
    st.set_page_config(
        page_title="TekaRx Priority Check",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Initialize session state
    if "selected_drugs" not in st.session_state:
        st.session_state["selected_drugs"] = []
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None

    _render_css()
    _render_header()

    # Load artifacts
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error(
            f"Failed to load model artifacts. "
            f"Build them with:\n{BUILD_COMMANDS}\n\nError: {exc}"
        )
        return

    bundle = artifacts["bundle"]
    dictionary = artifacts["dictionary"]
    threshold = _threshold(bundle)

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(
        ["New Check", "Results & Contributors", "Mapping Coverage", "About"]
    )

    with tab1:
        st.subheader("Patient Regimen")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            age = st.number_input(
                "Patient Age (years)",
                min_value=0,
                max_value=120,
                value=65,
                step=1,
            )
            sex = st.selectbox(
                "Biological Sex",
                ("Female", "Male", "Unknown"),
                index=0,
            )

        with col2:
            st.write("")  # Spacer for alignment

        st.markdown("---")

        # Drug autocomplete
        _drug_autocomplete(dictionary)

        # Selected drugs
        if st.session_state["selected_drugs"]:
            st.markdown('<div class="drug-chips">', unsafe_allow_html=True)
            st.write("**Selected medications:**")
            drugs = st.session_state.get("selected_drugs", [])
            for i, drug in enumerate(drugs):
                c1, c2 = st.columns([0.95, 0.05], gap="small")
                with c1:
                    st.write(f"🔹 {drug}")
                with c2:
                    if st.button("✕", key=f"del_{drug}_{i}"):
                        st.session_state["selected_drugs"].remove(drug)
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Run score button
        if st.button(
            "Run TekaRx Score",
            use_container_width=True,
            type="primary",
        ):
            if not st.session_state["selected_drugs"]:
                st.error("Please add at least one medication.")
            else:
                with st.spinner("Analyzing regimen..."):
                    matched, match_details = _match_medications(
                        st.session_state["selected_drugs"], dictionary
                    )
                    frame, summary = _build_features(age, sex, matched, bundle)
                    probability = _predict(frame, artifacts)
                    contributors = _top_contributors(bundle, frame)

                    st.session_state["last_prediction"] = {
                        "probability": probability,
                        "threshold": threshold,
                        "matched": matched,
                        "match_details": match_details,
                        "frame": frame,
                        "summary": summary,
                        "contributors": contributors,
                        "age": age,
                        "sex": sex,
                    }

                st.success("✓ Analysis complete! See Results & Contributors tab.")
                st.info("Switch to the Results tab to view detailed findings.")

    with tab2:
        if not st.session_state["last_prediction"]:
            st.markdown(
                """
                <div class="empty-state">
                  <h3>No Results Yet</h3>
                  <p>Run an analysis from the "New Check" tab to see results.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            pred = st.session_state["last_prediction"]
            probability = pred["probability"]
            is_high = probability >= pred["threshold"]
            label = "Review Priority Signal" if is_high else "No Priority Signal"

            st.markdown(
                f"""
                <div class="score-section">
                  <p style="margin: 0 0 0.5rem; color: var(--ink-light);
                    font-size: 0.85rem; text-transform: uppercase;
                    letter-spacing: 0.03em;">Model Priority Score</p>
                  <p class="score-value {'high' if is_high else ''}">
                    {probability:.0%}
                  </p>
                  <div class="priority-badge {'high' if is_high else ''}">
                    {label}
                  </div>
                  <p class="threshold-note">
                    Operating threshold: {pred['threshold']:.0%}
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Guidance
            if is_high:
                meaning = (
                    "Model priority signal detected. Review medications, "
                    "interactions, and top contributors below."
                )
                next_steps = (
                    "Conduct clinical review focusing on drug interactions "
                    "and cumulative toxicities."
                )
            else:
                meaning = "No priority signal under the trained IMRAD model."
                next_steps = (
                    "Continue standard monitoring; reassess if medications "
                    "change or new symptoms develop."
                )

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown(
                    f"""
                    <div class="guidance-card">
                      <h4>What This Means</h4>
                      <p>{meaning}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="guidance-card">
                      <h4>Next Steps</h4>
                      <p>{next_steps}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Status summary
            st.markdown("### Summary")
            st.markdown('<div class="status-grid">', unsafe_allow_html=True)
            for label, value in [
                ("Matched", pred["summary"]["Matched medications"]),
                ("Highest ROR", f"{pred['summary']['Highest ROR']:.2f}"),
                ("Boxed Warning", pred["summary"]["Boxed warning"]),
            ]:
                st.markdown(
                    f"""
                    <div class="status-item">
                      <div class="status-label">{label}</div>
                      <div class="status-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # Top contributors
            st.markdown("### Top 5 Contributing Factors")
            st.dataframe(
                pred["contributors"],
                hide_index=True,
                use_container_width=True,
            )

            st.markdown(
                f'<div class="disclaimer">{DISCLAIMER}</div>',
                unsafe_allow_html=True,
            )

    with tab3:
        if not st.session_state["last_prediction"]:
            st.markdown(
                """
                <div class="empty-state">
                  <h3>No Mapping Data Yet</h3>
                  <p>Run an analysis to see medication matching results.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            pred = st.session_state["last_prediction"]
            match_table = pd.DataFrame(pred["match_details"]).rename(
                columns={
                    "input": "Input",
                    "status": "Status",
                    "match": "Matched Name",
                    "score": "Confidence %",
                    "suggestions": "Suggestions",
                    "used": "Used",
                }
            )
            match_table["Confidence %"] = match_table["Confidence %"].map(
                lambda v: f"{v:.0f}%"
            )
            match_table["Status"] = match_table["Status"].map(
                lambda s: (
                    "Matched"
                    if s == "matched"
                    else ("Ambiguous" if s == "ambiguous" else "Unmatched")
                )
            )
            match_table["Used"] = match_table["Used"].map(lambda u: "Yes" if u else "No")

            st.markdown("### Medication Mapping Coverage")
            st.dataframe(match_table, hide_index=True, use_container_width=True)

            unmatched = sum(
                1 for m in pred["match_details"] if m["status"] == "unmatched"
            )
            if unmatched > 0:
                st.warning(
                    f"{unmatched} medication(s) not matched (< 90%). "
                    "Review suggestions or verify spellings."
                )

    with tab4:
        st.markdown(
            """
            ### How It Works

            TekaRx uses a frozen Random Forest model trained on IMRAD
            (Integrated Medication Risk Assessment) data to evaluate medication
            regimens. The model scores combinations based on:

            - **Drug characteristics**: ROR (Relative Odds Ratio) and ATC
              classifications
            - **Demographics**: Age and sex
            - **Complexity**: Polypharmacy patterns and therapeutic overlaps

            ### Key Limitations

            - Free-text matching is approximate; unmatched drugs get suggestions
            - Dosage, route, and reaction are not collected
            - Model reflects training data patterns, not causality
            - Results are research support, not medical diagnosis

            ### Safety & Disclaimer

            **{DISCLAIMER}**

            This tool is for research and decision support only. Always
            consult clinical judgment and comprehensive patient assessment
            before making treatment changes.

            ### Learn More

            - [TekaRx Documentation](https://github.com/tekarx)
            - [IMRAD Model Details](https://github.com/tekarx)
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
