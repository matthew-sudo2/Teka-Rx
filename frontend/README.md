# TekaRx frontend tools

## Streamlit medication priority dashboard

The visitor-facing Streamlit dashboard loads the actual trained model bundle from
`data/processed/models/imrad_models.joblib`. It does not retrain models or fabricate
predictions.

```powershell
python -m pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

The dashboard has one primary workflow: enter age, sex, and medication names, then run the
TekaRx Model Priority Score. Medication matching uses RapidFuzz against
`drug_dictionary.parquet["faers_raw"]`, and the result shows mapping coverage, the frozen
threshold, plain-language next steps, and the top five contributors.

See [the dashboard guide](../docs/streamlit_dashboard.md) for the artifact contract and
limitations.

## React prototype

This directory also contains the static, frontend-only TekaRx prototype. It includes a public
landing page, a Patient Notebooks directory, and a notebook-driven Patient Overview.

It intentionally has no backend, API calls, authentication, database, model inference, or
persistent patient state. The patient records are synthetic demonstration fixtures.

## Run locally

```powershell
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173/>.

## Production build

```powershell
npm run build
npm run preview
```
