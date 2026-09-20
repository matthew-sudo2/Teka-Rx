"""Visitor-facing Streamlit dashboard for the frozen TekaRx IMRAD models."""

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
        raise ValueError(f"Model bundle key {MODEL_KEY!r} does not expose predict_proba")


def _validate_dictionary(dictionary: pd.DataFrame) -> None:
    required = {"faers_raw", "dc_id", "atc_code", "ror", "has_boxed_warning"}
    missing = sorted(required - set(dictionary.columns))
    if missing:
        raise ValueError(f"Drug dictionary is missing columns: {', '.join(missing)}")


def _parse_medications(value: str) -> list[str]:
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def _match_medications(
    queries: list[str], dictionary: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    choices = dictionary["faers_raw"].dropna().drop_duplicates().astype(str).tolist()
    rows: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []

    for query in queries:
        candidates = process.extract(query, choices, scorer=fuzz.WRatio, limit=3)
        if not candidates:
            matches.append(
                {
                    "input": query,
                    "status": "unmatched",
                    "match": "",
                    "score": 0.0,
                    "suggestions": "No close match found",
                    "used": False,
                }
            )
            continue

        match, score, _ = candidates[0]
        status = _match_status(float(score))
        used = status == "matched"
        suggestions = ", ".join(str(candidate[0]) for candidate in candidates[:3])
        if used:
            rows.extend(dictionary.loc[dictionary["faers_raw"] == match].to_dict("records"))
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
        rors = pd.to_numeric(matched["ror"], errors="coerce").fillna(1.0).to_numpy()
        atc_codes = [
            code.strip().upper()
            for raw in matched["atc_code"].dropna().astype(str)
            for code in raw.split("|")
            if len(code.strip()) >= 3 and code.strip()[0].isalpha()
        ]
        boxed = float(pd.to_numeric(matched["has_boxed_warning"], errors="coerce").fillna(0).max())

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
                sum(code[:4] in {"N02A", "B01A", "M01A", "A10A"} for code in unique_atc)
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
        [{column: values.get(column, np.nan) for column in columns}], columns=columns
    )
    summary = {
        "Matched medications": num_drugs,
        "Highest ROR": values["max_ror"],
        "Average ROR": float(np.exp(values["mean_log_ror"])),
        "High-ROR count": int(values["high_ror_count"]),
        "ATC diversity": int(values["atc_diversity"]),
        "High-priority ATC groups": int(values["num_high_risk_atc"]),
        "Boxed warning": "Present" if boxed > 0 else "None",
    }
    return frame, summary


def _predict(frame: pd.DataFrame, artifacts: dict[str, Any]) -> float:
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
        for key in (MODEL_KEY, MODEL_NAME, "default"):
            value = thresholds.get(key)
            if isinstance(value, int | float):
                return float(value)
    return DEFAULT_THRESHOLD


def _top_contributors(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
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
                "Regimen Value": _format_feature_value(col, val),
                "Relative Importance": float(imp),
            }
        )

    df = pd.DataFrame(records).sort_values("Relative Importance", ascending=False).head(5)
    total_top = df["Relative Importance"].sum()
    if total_top > 0:
        df["Relative Contribution"] = df["Relative Importance"].map(lambda v: f"{v / total_top:.1%}")
    else:
        df["Relative Contribution"] = "N/A"
    return df[["Feature", "Regimen Value", "Relative Contribution"]]


def _feature_label(name: str) -> str:
    labels = {
        "max_ror": "Highest Drug ROR",
        "mean_log_ror": "Average Log ROR",
        "high_ror_count": "High-ROR Medication Count",
        "has_boxed_warning": "Boxed Warning Present",
        "num_drugs": "Matched Medication Count",
        "num_drugs_squared": "Medication Count (Squared)",
        "polypharmacy_age": "Polypharmacy × Age Interaction",
        "atc_diversity": "ATC Category Diversity",
        "atc_l2_diversity": "ATC Level-2 Diversity",
        "atc_l3_diversity": "ATC Level-3 Diversity",
        "num_high_risk_atc": "High-Risk ATC Count (CNS/Cardio/Blood)",
        "num_high_risk_atc_groups": "High-Risk ATC Subgroups",
        "therapeutic_duplicates": "Therapeutic Duplicates (Level 1)",
        "age_imputed_years": "Patient Age",
        "age_group_65_plus": "Age 65 or Older",
        "age_group_41_64": "Age 41 to 64",
        "age_group_18_40": "Age 18 to 40",
        "sex_unknown": "Sex Unspecified",
        "patient_avg_cluster_risk": "Graph Cluster Relational Risk (Avg)",
        "patient_max_cluster_risk": "Graph Cluster Relational Risk (Max)",
        "patient_avg_propagated_risk": "Label Propagated Relational Risk (Avg)",
        "patient_max_propagated_risk": "Label Propagated Relational Risk (Max)",
        "patient_avg_neighbor_ror": "Relational Neighbor ROR (Avg)",
        "patient_max_neighbor_ror": "Relational Neighbor ROR (Max)",
        "patient_avg_drug_degree": "Graph Drug Co-exposure Degree (Avg)",
        "patient_max_drug_degree": "Graph Drug Co-exposure Degree (Max)",
    }
    if name.startswith("atc_l1_count_"):
        letter = name.split("_")[-1].upper()
        return f"ATC Class {letter} Medication Count"
    return labels.get(name, name.replace("_", " ").title())


def _format_feature_value(name: str, val: Any) -> str:
    if val is None or pd.isna(val):
        return "Not documented"
    val = float(val)
    if "has_" in name or "_missing" in name or "_unknown" in name or name.startswith("age_group_"):
        return "Yes" if val >= 0.5 else "No"
    if name in {"num_drugs", "high_ror_count", "atc_diversity", "atc_l2_diversity", "therapeutic_duplicates"}:
        return f"{int(round(val))}"
    if name == "age_imputed_years":
        return f"{val:.0f} years"
    if name == "polypharmacy_age":
        return f"{val:.1f}"
    if "ror" in name:
        return f"{val:.2f}"
    return f"{val:.2f}"


def _priority_label(probability: float, threshold: float) -> str:
    return "Review Priority Signal" if probability >= threshold else "No Priority Signal"


def _result_guidance(probability: float, threshold: float, unmatched_count: int) -> tuple[str, str]:
    if probability >= threshold:
        meaning = (
            "The model priority score is at or above the frozen operating threshold (50%). "
            "Based on the patient demographics and pharmacological features of the reported regimen, "
            "this combination exhibits a priority signal consistent with serious adverse event patterns."
        )
        next_steps = (
            "Conduct a clinical review of the medication list, paying particular attention to drug interactions, "
            "cumulative organ toxicities, and the top contributing factors surfaced below."
        )
    else:
        meaning = (
            "The model priority score is below the frozen operating threshold (50%). "
            "The combination presents No Priority Signal under the trained IMRAD seriousness model."
        )
        next_steps = (
            "Document the current regimen and continue standard monitoring. Re-evaluate if medications are "
            "added, doses adjusted, or if new symptoms arise."
        )

    if unmatched_count > 0:
        noun = "medication was" if unmatched_count == 1 else "medications were"
        next_steps = (
            f"{next_steps} Note: {unmatched_count} {noun} not matched in the clinical dictionary. "
            "Verify spellings using the suggestions below to ensure complete model coverage."
        )
    return meaning, next_steps


def _show_missing_artifacts(status: ArtifactStatus, error: Exception | None = None) -> None:
    missing_list = "\n".join(f"- {path.relative_to(ROOT)}" for path in status.missing)
    st.markdown(
        f"""
        <section class="artifact-panel" role="alert">
          <h2>Model artifacts are not ready yet</h2>
          <p>This clinical dashboard runs strictly with the trained TekaRx model bundle and reference dictionary.
          Models are never retrained inside the interface.</p>
          <p class="disclaimer"><strong>Disclaimer:</strong> {DISCLAIMER}</p>
          <p><strong>Missing artifact files:</strong></p>
          <pre>{missing_list or "Artifact validation failed"}</pre>
          <p><strong>Build them from the repository root:</strong></p>
          <pre>{BUILD_COMMANDS}</pre>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if error is not None and not status.missing:
        st.caption(f"Validation detail: {error}")


def _render_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #16221c;
          --ink-secondary: #485750;
          --border: #d8e2dc;
          --panel-bg: #f8faf9;
          --panel-border: #e3ece6;
          --green-primary: #1e5b3a;
          --green-soft: #eaf3ed;
          --green-border: #c3ddcc;
          --amber-primary: #8a580a;
          --amber-soft: #fef6e7;
          --amber-border: #f2dcab;
          --red-primary: #9b2828;
          --red-soft: #fbeeed;
          --white: #ffffff;
        }

        /* App base */
        .stApp {
          background-color: var(--white);
          color: var(--ink);
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        .block-container {
          max-width: 1140px;
          padding-top: 2rem;
          padding-bottom: 3.5rem;
        }

        /* Typography */
        h1 {
          color: var(--ink);
          font-size: clamp(1.8rem, 3.5vw, 2.6rem);
          font-weight: 700;
          line-height: 1.15;
          margin-bottom: 0.35rem;
          letter-spacing: -0.02em;
        }

        h2 {
          color: var(--ink);
          font-size: 1.25rem;
          font-weight: 650;
          margin-top: 0.2rem;
          margin-bottom: 0.6rem;
          letter-spacing: -0.01em;
        }

        h3 {
          color: var(--ink);
          font-size: 1.05rem;
          font-weight: 600;
          margin-bottom: 0.4rem;
        }

        p, label, .stMarkdown, .stCaption {
          color: var(--ink);
        }

        /* Header hero */
        .hero {
          border-bottom: 1px solid var(--border);
          margin-bottom: 1.6rem;
          padding-bottom: 1.1rem;
        }

        .hero p {
          color: var(--ink-secondary);
          font-size: 1.05rem;
          line-height: 1.5;
          max-width: 780px;
          margin-top: 0.3rem;
        }

        /* Form container */
        div[data-testid="stForm"] {
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 1.3rem;
          background: var(--white);
          box-shadow: 0 1px 3px rgba(22, 34, 28, 0.04);
        }

        /* Result container */
        .result-panel {
          border: 1px solid var(--panel-border);
          border-radius: 8px;
          padding: 1.35rem;
          background: var(--panel-bg);
          margin-bottom: 1rem;
        }

        .result-panel h2 {
          font-size: 1.05rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--ink-secondary);
          margin-bottom: 0.2rem;
        }

        .score-row {
          display: flex;
          align-items: baseline;
          gap: 1rem;
          margin: 0.4rem 0 0.8rem;
          flex-wrap: wrap;
        }

        .score-value {
          font-size: clamp(2.8rem, 6vw, 4.4rem);
          line-height: 1;
          font-weight: 750;
          color: var(--green-primary);
          letter-spacing: -0.03em;
        }

        .score-value.high {
          color: var(--amber-primary);
        }

        .priority-badge {
          display: inline-flex;
          align-items: center;
          border-radius: 6px;
          padding: 0.4rem 0.75rem;
          font-size: 0.95rem;
          font-weight: 650;
          background: var(--green-soft);
          color: var(--green-primary);
          border: 1px solid var(--green-border);
        }

        .priority-badge.high {
          background: var(--amber-soft);
          color: var(--amber-primary);
          border: 1px solid var(--amber-border);
        }

        .threshold-caption {
          color: var(--ink-secondary);
          font-size: 0.92rem;
          margin-bottom: 0.75rem;
        }

        .guidance-block {
          margin-top: 0.75rem;
          padding-top: 0.75rem;
          border-top: 1px solid var(--panel-border);
          font-size: 0.95rem;
          line-height: 1.5;
        }

        .guidance-block strong {
          color: var(--ink);
        }

        /* Summary status metrics */
        .status-grid {
          display: grid;
          gap: 0.75rem;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          margin: 0.9rem 0 1.2rem;
        }

        .status-card {
          border: 1px solid var(--border);
          border-radius: 7px;
          padding: 0.75rem 0.85rem;
          background: var(--panel-bg);
        }

        .status-card span {
          display: block;
          font-size: 0.82rem;
          color: var(--ink-secondary);
          text-transform: uppercase;
          letter-spacing: 0.03em;
          margin-bottom: 0.2rem;
        }

        .status-card strong {
          display: block;
          font-size: 1.25rem;
          color: var(--ink);
          font-weight: 650;
        }

        /* Empty state prompt */
        .empty-panel {
          border: 1px dashed var(--border);
          border-radius: 8px;
          padding: 1.8rem 1.4rem;
          background: var(--panel-bg);
          text-align: center;
        }

        .empty-panel h3 {
          font-size: 1.15rem;
          color: var(--ink);
          margin-bottom: 0.4rem;
        }

        .empty-panel p {
          color: var(--ink-secondary);
          font-size: 0.95rem;
          max-width: 480px;
          margin: 0 auto 1.2rem;
        }

        /* Disclaimers */
        .disclaimer {
          color: var(--ink-secondary);
          border-left: 3px solid var(--green-primary);
          padding-left: 0.65rem;
          font-size: 0.88rem;
          line-height: 1.4;
          margin-top: 0.85rem;
        }

        /* Artifact error panel */
        .artifact-panel {
          border: 1px solid var(--amber-border);
          border-radius: 8px;
          padding: 1.3rem;
          background: var(--amber-soft);
          margin-bottom: 1.5rem;
        }

        .artifact-panel h2 {
          color: var(--amber-primary);
          margin-top: 0;
        }

        .artifact-panel pre {
          white-space: pre-wrap;
          background: var(--white);
          border: 1px solid var(--amber-border);
          border-radius: 6px;
          padding: 0.8rem;
          overflow-x: auto;
          font-size: 0.88rem;
        }

        /* Button styles */
        .stButton > button {
          border-radius: 6px;
          border: 1px solid var(--green-primary);
          background: var(--green-primary);
          color: var(--white);
          font-weight: 600;
          font-size: 0.98rem;
          min-height: 2.85rem;
          transition: background 0.15s ease-in-out;
        }

        .stButton > button:hover {
          background: #184a2f;
          border-color: #184a2f;
          color: var(--white);
        }

        .stButton > button:focus {
          outline: 3px solid #b7dbca;
          outline-offset: 2px;
        }

        /* Secondary example button styling */
        button[kind="secondary"] {
          background: var(--white) !important;
          color: var(--green-primary) !important;
          border: 1px solid var(--border) !important;
          min-height: 2.2rem !important;
          font-size: 0.88rem !important;
        }

        button[kind="secondary"]:hover {
          background: var(--green-soft) !important;
          border-color: var(--green-primary) !important;
        }

        /* Dataframe tables */
        div[data-testid="stDataFrame"] {
          border: 1px solid var(--border);
          border-radius: 7px;
          overflow: hidden;
          margin-top: 0.4rem;
        }

        /* Responsive layout */
        @media (max-width: 768px) {
          .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
          }
          .status-grid {
            grid-template-columns: 1fr;
          }
          .score-row {
            gap: 0.5rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <header class="hero">
          <h1>TekaRx Medication Priority Check</h1>
          <p>Clinical decision-support tool evaluating medication regimens against the frozen TekaRx
          IMRAD seriousness classification models. Enter patient age, sex, and medications to inspect the
          Model Priority Score, medication mapping coverage, and top predictive contributors.</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-panel">
          <h3>No Regimen Evaluated Yet</h3>
          <p>Enter patient details and medications in the form on the left, or load one of the clinical
          test regimens below to inspect the model output.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load High-Priority Polypharmacy Example", key="btn_high_example", use_container_width=True):
            st.session_state["prefill_age"] = 74
            st.session_state["prefill_sex"] = "Female"
            st.session_state["prefill_meds"] = "Warfarin\nIbuprofen\nTramadol\nMetoprolol\nOmeprazole"
            st.session_state["auto_submit"] = True
            st.rerun()

    with col2:
        if st.button("Load Maintenance Monotherapy Example", key="btn_low_example", use_container_width=True):
            st.session_state["prefill_age"] = 52
            st.session_state["prefill_sex"] = "Male"
            st.session_state["prefill_meds"] = "Metformin\nAtorvastatin\nLevothyroxine"
            st.session_state["auto_submit"] = True
            st.rerun()


def _render_match_feedback(match_details: list[dict[str, Any]]) -> int:
    if not match_details:
        return 0

    table = pd.DataFrame(match_details)
    unmatched_count = int((table["status"] != "matched").sum())

    display = table.rename(
        columns={
            "input": "Entered Medication",
            "status": "Mapping Status",
            "match": "Matched Dictionary Name",
            "score": "Match Confidence",
            "suggestions": "Suggested RapidFuzz Correction",
            "used": "Included in Model Features",
        }
    )
    display["Match Confidence"] = display["Match Confidence"].map(lambda v: f"{v:.0f}%")
    display["Mapping Status"] = display["Mapping Status"].map(
        lambda s: "Matched" if s == "matched" else ("Ambiguous" if s == "ambiguous" else "Unmatched")
    )
    display["Included in Model Features"] = display["Included in Model Features"].map(
        lambda u: "Yes" if u else "No"
    )

    st.subheader("Medication Mapping Coverage")
    st.dataframe(display, hide_index=True, use_container_width=True)

    if unmatched_count > 0:
        st.warning(
            f"{unmatched_count} medication name(s) could not be mapped with >= 90% confidence. "
            "Please review the suggestions above or confirm standard clinical generic spellings."
        )
    return unmatched_count


def _render_result(
    *,
    probability: float,
    threshold: float,
    unmatched_count: int,
    contributors: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    label = _priority_label(probability, threshold)
    is_high = probability >= threshold
    score_class = "high" if is_high else ""
    meaning, next_steps = _result_guidance(probability, threshold, unmatched_count)

    st.markdown(
        f"""
        <section class="result-panel" aria-live="polite">
          <h2>Model Priority Score</h2>
          <div class="score-row">
            <div class="score-value {score_class}">{probability:.0%}</div>
            <div class="priority-badge {score_class}">{label}</div>
          </div>
          <div class="threshold-caption">Operating decision threshold: {threshold:.0%} | Trained Random Forest paradigm</div>
          <div class="guidance-block">
            <p><strong>What this means:</strong> {meaning}</p>
            <p><strong>Next steps:</strong> {next_steps}</p>
            <p class="disclaimer">{DISCLAIMER}</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-grid">
          <div class="status-card">
            <span>Matched Medications</span>
            <strong>{summary['Matched medications']}</strong>
          </div>
          <div class="status-card">
            <span>Highest Drug ROR</span>
            <strong>{summary['Highest ROR']:.2f}</strong>
          </div>
          <div class="status-card">
            <span>Boxed Warning</span>
            <strong>{summary['Boxed warning']}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Contribution to Model Score")
    st.caption("Top feature contributions from the frozen model pipeline for this specific patient input.")
    st.dataframe(contributors, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="TekaRx Medication Priority Check",
        page_icon="Rx",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _render_css()
    _render_header()

    status = _artifact_status()
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        _show_missing_artifacts(status, exc)
        return

    bundle = artifacts["bundle"]
    threshold = _threshold(bundle)

    # Handle preset examples via session state
    prefill_age = st.session_state.get("prefill_age", 65)
    prefill_sex = st.session_state.get("prefill_sex", "Female")
    prefill_meds = st.session_state.get("prefill_meds", "")
    sex_options = ("Female", "Male", "Unknown")
    sex_index = sex_options.index(prefill_sex) if prefill_sex in sex_options else 2

    auto_submit = st.session_state.pop("auto_submit", False)

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        with st.form("prediction_form"):
            st.subheader("Patient Regimen Entry")
            age = st.number_input(
                "Patient Age (years)",
                min_value=0,
                max_value=120,
                value=int(prefill_age),
                step=1,
                help="Patient chronological age in years.",
            )
            sex = st.selectbox(
                "Biological Sex",
                sex_options,
                index=sex_index,
                help="Reported patient sex.",
            )
            medications = st.text_area(
                "Active Medications",
                value=prefill_meds,
                height=180,
                placeholder="Example:\nAspirin\nMetformin\nAtorvastatin\nLisinopril",
                help="Enter one medication per line, or separate drug names with commas.",
            )
            submitted = st.form_submit_button("Run TekaRx score", use_container_width=True)

        st.caption(
            f"{DISCLAIMER} Frozen model evaluates {len(bundle['feature_cols'])} relational and pharmacological features."
        )

    queries = _parse_medications(medications)
    trigger_run = submitted or auto_submit

    with right:
        if not trigger_run and not queries:
            _render_empty_state()
            return

        if trigger_run and not queries:
            st.error("Please enter at least one medication name before running the evaluation.")
            _render_empty_state()
            return

        with st.spinner("Mapping medications and executing frozen TekaRx pipeline..."):
            matched, match_details = _match_medications(queries, artifacts["dictionary"])
            frame, summary = _build_features(age, sex, matched, bundle)
            probability = _predict(frame, artifacts)
            contributors = _top_contributors(bundle, frame)

        unmatched_count = _render_match_feedback(match_details)
        _render_result(
            probability=probability,
            threshold=threshold,
            unmatched_count=unmatched_count,
            contributors=contributors,
            summary=summary,
        )


if __name__ == "__main__":
    main()
