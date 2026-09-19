# TekaRx frontend tools

## Streamlit verification dashboard

The clinician-facing verification dashboard loads the actual trained models from
`data/processed/models/imrad_models.joblib`. It does not retrain or fabricate predictions.

```powershell
python -m pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

The three pages are Patient Risk Triage, Model Comparison, and Batch Verification. The first
page accepts age, sex, weight, and medication names; the second compares all three classifiers;
the third accepts a feature-aligned CSV and can recompute the stored test-set AUC.

To add an input feature, add its model-column name to the feature-building logic in
`streamlit_app.py`, keep the name and order in the saved bundle authoritative, and add a focused
batch test. Medication matching uses RapidFuzz against `drug_dictionary.parquet["faers_raw"]`;
the sidebar shows the selected match and score so mappings can be reviewed.

See [the dashboard guide](../docs/streamlit_dashboard.md) for the artifact contract,
verification workflow, and limitations.

## React prototype

This directory contains the static, frontend-only TekaRx prototype. It includes a public landing page, a Patient Notebooks directory, and a notebook-driven Patient Overview.

It intentionally has no backend, API calls, authentication, database, model inference, or persistent patient state. The patient records are synthetic demonstration fixtures. Hash navigation connects the two prototype views; clinical action controls remain presentational.

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

## Visual implementation

- The shared application shell uses muted forest-green glass surfaces, neutral gray-green borders, and semantic red, amber, and green status colors.
- The landing page is the default route and is also available at `#home`.
- Patient Overview is available at `#patient-overview`; Patient Notebooks is available at `#patient-notebooks`.
- Notebook illustrations are lightweight CSS components with a restrained green spine, gray binding rings, avatar, and document lines.
- The application background combines a near-black green gradient with a fixed monochromatic SVG-turbulence grain overlay.
- The grain is embedded in CSS at very low opacity and does not require an external image asset.
- Grain is disabled for print and forced-color modes.
- Space Grotesk is bundled locally through `@fontsource`; the page does not rely on a remote font request.
