"""Local Streamlit verification dashboard for the trained TekaRx models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
import streamlit as st
from rapidfuzz import fuzz, process
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DISCLAIMER = (
    "This is a research verification tool. Predictions are for research decision-support "
    "only and do not constitute medical advice."
)
MODEL_NAMES = ("Random Forest", "SVM", "AdaBoost")
MODEL_KEYS = {
    "Random Forest": "random_forest",
    "SVM": "svm_calibrated",
    "AdaBoost": "adaboost",
}
GRAPH_FEATURES = (
    "patient_avg_cluster_risk",
    "patient_avg_drug_degree",
    "patient_avg_neighbor_ror",
    "patient_avg_propagated_risk",
    "patient_max_cluster_risk",
    "patient_max_drug_degree",
    "patient_max_neighbor_ror",
    "patient_max_propagated_risk",
)


@st.cache_resource(show_spinner="Loading trained TekaRx artifacts...")
def load_artifacts() -> dict[str, Any]:
    """Load the persisted model bundle and small reference tables once per process."""
    bundle_path = PROCESSED / "models" / "imrad_models.joblib"
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Missing {bundle_path}. Run the IMRAD notebook training/save cell first."
        )
    bundle = joblib.load(bundle_path)
    dictionary_path = PROCESSED / "drug_dictionary.parquet"
    if not dictionary_path.is_file():
        raise FileNotFoundError(f"Missing drug dictionary: {dictionary_path}")
    dictionary = pd.read_parquet(dictionary_path)
    graph_path = PROCESSED / "tekarx_graph_features.parquet"
    graph_features = pd.read_parquet(graph_path) if graph_path.is_file() else None
    return {"bundle": bundle, "dictionary": dictionary, "graph_features": graph_features}


def _match_medications(
    queries: list[str], dictionary: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    choices = dictionary["faers_raw"].dropna().drop_duplicates().astype(str).tolist()
    other_rows = dictionary.loc[dictionary["dc_id"] == 0]
    rows: list[pd.Series] = []
    matches: list[dict[str, Any]] = []
    for query in queries:
        result = process.extractOne(query, choices, scorer=fuzz.WRatio)
        if result is None:
            matches.append({"input": query, "match": "OTHER", "score": 0.0})
            rows.extend(other_rows.to_dict("records"))
            continue
        matched_name, score, _ = result
        matched_rows = dictionary[dictionary["faers_raw"] == matched_name]
        rows.extend(matched_rows.to_dict("records"))
        matches.append({"input": query, "match": matched_name, "score": float(score)})
    if rows:
        return pd.DataFrame(rows), matches
    return dictionary.iloc[0:0].copy(), matches


def _feature_defaults(bundle: dict[str, Any]) -> dict[str, float]:
    values = bundle["imputer"].statistics_
    return dict(zip(bundle["feature_cols"], values, strict=True))


def _build_features(
    age: float,
    sex: str,
    weight: float | None,
    matched: pd.DataFrame,
    bundle: dict[str, Any],
    graph_features: pd.DataFrame | None,
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
            "weight_kg_normalized": min(max((weight or 70.0) / 250.0, 0.0), 1.0),
            "weight_missing": float(weight is None),
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
            "high_ror_count": float(np.sum(rors > 2)),
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
        values[f"atc_l1_count_{letter}"] = float(l1.count(letter.upper()))

    graph_source = "train-fitted imputer medians"
    if graph_features is not None:
        for name in GRAPH_FEATURES:
            if name in graph_features.columns:
                numeric = pd.to_numeric(graph_features[name], errors="coerce").dropna()
                if not numeric.empty:
                    values[name] = float(numeric.median())
                    graph_source = "graph feature lookup medians"
    frame = pd.DataFrame(
        [{column: values.get(column, np.nan) for column in columns}], columns=columns
    )
    summary = {
        "num_drugs": num_drugs,
        "max_ror": values["max_ror"],
        "average_ror": float(np.exp(values["mean_log_ror"])),
        "high_ror_count": values["high_ror_count"],
        "atc_diversity": values["atc_diversity"],
        "high_risk_atc_count": values["num_high_risk_atc"],
        "graph_source": graph_source,
    }
    return frame, summary


def _predict(frame: pd.DataFrame, model_name: str, artifacts: dict[str, Any]) -> float:
    bundle = artifacts["bundle"]
    transformed = bundle["scaler"].transform(bundle["imputer"].transform(frame))
    model = bundle[MODEL_KEYS[model_name]]
    return float(model.predict_proba(transformed)[0, 1])


def _explanation(model: Any, feature_cols: list[str], frame: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = np.abs(model.feature_importances_)
    elif hasattr(model, "calibrated_classifiers_"):
        coefficients = []
        for calibrated in model.calibrated_classifiers_:
            estimator = getattr(
                calibrated, "estimator", getattr(calibrated, "base_estimator", None)
            )
            if estimator is not None and hasattr(estimator, "coef_"):
                coefficients.append(np.abs(estimator.coef_[0]))
        importance = np.mean(coefficients, axis=0) if coefficients else np.zeros(len(feature_cols))
    else:
        importance = np.zeros(len(feature_cols))
    result = pd.DataFrame({"Feature": feature_cols, "Importance": importance})
    result["Value"] = frame.iloc[0].to_numpy()
    return result.sort_values("Importance", ascending=False).head(5)


def _metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    result = {
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
    }
    if y_true.nunique() > 1:
        result["AUC"] = roc_auc_score(y_true, probability)
        result["PR-AUC"] = average_precision_score(y_true, probability)
    return result


def _verify_model(artifacts: dict[str, Any], model_name: str) -> float:
    path = PROCESSED / "tekarx_cohort_enriched.parquet"
    required = artifacts["bundle"]["feature_cols"]
    available = set(parquet.ParquetFile(path).schema.names)
    columns = [column for column in required if column in available] + ["split", "is_serious"]
    test = pd.read_parquet(path, columns=columns, filters=[("split", "==", "test")])
    probability = _predict_batch(test, model_name, artifacts)
    return float(roc_auc_score(test["is_serious"], probability))


def _predict_batch(frame: pd.DataFrame, model_name: str, artifacts: dict[str, Any]) -> np.ndarray:
    bundle = artifacts["bundle"]
    aligned = frame.reindex(columns=bundle["feature_cols"])
    transformed = bundle["scaler"].transform(bundle["imputer"].transform(aligned))
    return bundle[MODEL_KEYS[model_name]].predict_proba(transformed)[:, 1]


def main() -> None:
    st.set_page_config(page_title="TekaRx model verification", page_icon="Rx", layout="wide")
    st.warning(DISCLAIMER)
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    bundle = artifacts["bundle"]
    page = st.sidebar.radio(
        "Page", ("Patient Risk Triage", "Model Comparison", "Batch Verification")
    )
    st.sidebar.caption(
        f"Loaded {len(bundle['feature_cols'])} model features from imrad_models.joblib"
    )

    if page in ("Patient Risk Triage", "Model Comparison"):
        st.title(page)
        left, right = st.columns(2)
        with left:
            st.subheader("Patient input")
            age = st.number_input("Age", 0, 120, 65)
            sex = st.selectbox("Sex", ("Male", "Female", "Unknown"))
            weight = st.number_input("Weight (kg)", 0.0, 300.0, value=None, step=0.1)
            medication_text = st.text_area(
                "Medications", placeholder="One medication per line or comma-separated"
            )
            st.multiselect("Indications", ("Malignancy", "Cardiovascular", "Infection", "Other"))
            queries = [
                item.strip()
                for item in medication_text.replace("\n", ",").split(",")
                if item.strip()
            ]
            matched, match_details = _match_medications(queries, artifacts["dictionary"])
            frame, summary = _build_features(
                age, sex, weight, matched, bundle, artifacts["graph_features"]
            )
            st.subheader("Auto-computed model features")
            st.dataframe(
                pd.DataFrame([summary]).T.rename(columns={0: "Value"}),
                use_container_width=True,
            )
        with right:
            st.subheader("Prediction output")
            model_name = st.radio("Model", MODEL_NAMES, horizontal=True)
            threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
            probability = _predict(frame, model_name, artifacts)
            label = "SERIOUS" if probability >= threshold else "NON-SERIOUS"
            st.metric("Predicted class", label)
            st.metric("Serious probability", f"{probability:.2f}")
            st.progress(probability, text=f"Model probability: {probability:.1%}")
            explanation = _explanation(
                bundle[MODEL_KEYS[model_name]], bundle["feature_cols"], frame
            )
            st.write("Top five model signals")
            st.dataframe(explanation, hide_index=True, use_container_width=True)
        st.sidebar.subheader("Medication matches")
        st.sidebar.dataframe(pd.DataFrame(match_details), hide_index=True, use_container_width=True)
        if summary["graph_source"] != "graph feature lookup medians":
            st.info(
                "The standalone graph feature lookup is unavailable; graph features use "
                "the bundle's train-fitted imputer medians."
            )
        if page == "Model Comparison":
            probabilities = {name: _predict(frame, name, artifacts) for name in MODEL_NAMES}
            comparison = pd.DataFrame(
                {
                    "Model": list(probabilities),
                    "Serious probability": list(probabilities.values()),
                }
            )
            comparison.loc[len(comparison)] = [
                "Ensemble average",
                np.mean(list(probabilities.values())),
            ]
            st.subheader("Same-input comparison")
            st.dataframe(comparison, hide_index=True, use_container_width=True)
            st.bar_chart(comparison.set_index("Model"))
    else:
        st.title("Batch Verification")
        uploaded = st.file_uploader(
            "Upload a CSV containing the exact model feature columns", type="csv"
        )
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="batch_threshold")
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            st.write(f"Loaded {len(batch):,} rows.")
            truth_column = "is_serious" if "is_serious" in batch.columns else None
            predictions = batch.copy()
            for model_name in MODEL_NAMES:
                probability = _predict_batch(batch, model_name, artifacts)
                predictions[f"{model_name.lower().replace(' ', '_')}_probability"] = probability
                predictions[
                    f"{model_name.lower().replace(' ', '_')}_prediction"
                ] = (probability >= threshold).astype(int)
            st.dataframe(predictions.head(100), use_container_width=True)
            if truth_column:
                model_metrics = {}
                for model_name in MODEL_NAMES:
                    key = f"{model_name.lower().replace(' ', '_')}_probability"
                    model_metrics[model_name] = _metrics(
                        batch[truth_column], predictions[key].to_numpy(), threshold
                    )
                st.dataframe(pd.DataFrame(model_metrics).T, use_container_width=True)
        if st.button("Verify against the stored test split"):
            with st.spinner("Recomputing test-set predictions..."):
                auc = _verify_model(artifacts, "Random Forest")
            st.success(f"Model verified - recomputed Random Forest test AUC: {auc:.4f}")


if __name__ == "__main__":
    main()