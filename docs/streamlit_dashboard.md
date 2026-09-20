# Streamlit dashboard

## Purpose

The Streamlit app is a visitor-facing medication priority check for the trained TekaRx
IMRAD model bundle. It loads `data/processed/models/imrad_models.joblib`, uses the stored
`feature_cols` order, then runs bundle imputer, bundle scaler, and `predict_proba`. It does
not retrain models or fabricate unavailable inputs. Results are research decision-support
signals only, not medical advice.

## Run locally

From the repository root:

```powershell
python -m pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

The app expects:

- `data/processed/models/imrad_models.joblib`
- `data/processed/drug_dictionary.parquet`

If either artifact is missing, the app shows an in-page recovery panel with the repository
commands to rebuild the processed data and save the model bundle.

```powershell
python -m pip install -e .[imrad,notebook]
tekarx build-prospective --data-dir data
jupyter nbconvert --to notebook --execute notebooks/TekaRx_IMRAD_Models.ipynb --output TekaRx_IMRAD_Models.executed.ipynb --output-dir notebooks
```

## Visitor Workflow

The dashboard is built around one task: enter age, sex, and medications, then run the TekaRx
score. Medication names are fuzzy-matched with RapidFuzz against
`drug_dictionary.parquet["faers_raw"]`. The result includes:

- Model Priority Score with the frozen threshold
- No Priority Signal or Review Priority Signal label
- Medication Mapping Coverage for matched, ambiguous, and unmatched names
- Contribution to Model Score, using the top five model feature importances
- The one-line research decision-support disclaimer beside the result

The app initializes the feature row from the train-fitted imputer statistics and only replaces
fields that can be derived from the visitor form and dictionary mapping. This keeps the saved
feature schema authoritative and prevents accidental column reordering.

## Reproduce a Prediction from Python

```python
import joblib
import pandas as pd

artifacts = joblib.load("data/processed/models/imrad_models.joblib")
frame = pd.DataFrame([{column: None for column in artifacts["feature_cols"]}])
X = artifacts["scaler"].transform(artifacts["imputer"].transform(frame))
probability = artifacts["random_forest"].predict_proba(X)[:, 1]
```

The dashboard follows the same model path: bundle imputer, bundle scaler, then
`predict_proba`.

## Known Limitations

- Free-text medication matching is approximate. Confirm ambiguous and unmatched names.
- Dosage, reaction, route, and patient-specific graph inputs are not collected in the form, so
  unavailable values remain train-fitted imputer defaults.
- Model probabilities reflect the training data and calibration. They do not establish drug
  causality or patient-specific diagnosis.
