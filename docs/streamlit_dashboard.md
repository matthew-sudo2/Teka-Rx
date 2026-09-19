# Streamlit dashboard

## Purpose

The Streamlit app is a local clinician-facing verification tool for the trained TekaRx
classifiers. It loads the persisted `data/processed/models/imrad_models.joblib` bundle and
does not retrain models. Results are research decision-support signals only, not diagnoses or
medical advice.

## Run locally

From the repository root:

```powershell
python -m pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

The app expects the model bundle, `drug_dictionary.parquet`, and
`tekarx_cohort_enriched.parquet` under `data/processed/`. The model bundle is the artifact
written by the final save cell in `notebooks/TekaRx_IMRAD_Models.ipynb`.

## Pages

### Patient Risk Triage

Enter age, sex, optional weight, and comma- or newline-separated medication names. Each name is
matched against `drug_dictionary.parquet` with RapidFuzz, and the sidebar shows the selected raw
name and match score. The page displays the derived drug/ATC/ROR features, model probability,
thresholded class, and the five highest model importance/coefficient signals.

The app aligns the input to the exact 87-column `feature_cols` list stored in the joblib bundle.
Features unavailable from a free-text clinical form, including dosage and patient-specific graph
statistics, are initialized from the train-fitted imputer. This prevents accidental column
reordering or fabricated values.

### Model Comparison

The same feature row is sent through Random Forest, calibrated SVM, and AdaBoost. The page shows
the three serious probabilities, their ensemble average, and a bar chart.

### Batch Verification

Upload a CSV containing the exact model feature columns. The app computes all three probabilities
and classes. If the CSV contains `is_serious`, it reports AUC, PR-AUC, accuracy, precision,
recall, and F1. The **Verify against the stored test split** button reads the held-out test rows
from the enriched cohort and recomputes a Random Forest AUC from the persisted artifacts.

## Reproduce a prediction from the notebook

```python
import joblib
import pandas as pd

artifacts = joblib.load("data/processed/models/imrad_models.joblib")
features = pd.read_parquet("data/processed/tekarx_cohort_enriched.parquet")
feature_cols = artifacts["feature_cols"]
test = features.loc[features["split"] == "test"]
X = artifacts["scaler"].transform(artifacts["imputer"].transform(test[feature_cols]))
probability = artifacts["random_forest"].predict_proba(X)[:, 1]
```

The app follows the same order: bundle imputer, bundle scaler, then `predict_proba`.

## Screenshots

Screenshots can be added here after a local run. Suggested captures are the triage page with
match scores visible, the model comparison chart, and the batch verification metrics table.

## Known limitations

- The app is a verification surface, not a deployment system or clinical decision system.
- The current repository contains no standalone `tekarx_graph_features.parquet` at the root.
  When it is present, the app reads its graph columns; otherwise it uses the trained imputer
  medians and displays that fallback explicitly.
- Free-text medication matching is approximate. A high fuzzy score is not evidence of clinical
  equivalence, and unmatched names fall back to baseline dictionary behavior.
- Dosage, reactions, route, and other exposure features cannot be recovered reliably from the
  small manual form, so they remain train-fitted defaults.
- Batch CSVs must use the exact feature names and semantics from the saved bundle.
- Model probabilities reflect the training data and calibration; they do not establish drug
  causality or patient-specific risk.