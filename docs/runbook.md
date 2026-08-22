# Extraction runbook

For the fixed `gnn-full` Google Colab workflow, including Drive/local-disk separation,
restart markers, CUDA training, and locked-test checks, see [colab.md](colab.md) and the
[`train_full_colab.ipynb`](../notebooks/train_full_colab.ipynb) notebook.

This guide starts from a fresh Windows PowerShell session in the TekaRx repository. Source
downloads can be large, so begin with one FAERS quarter and the compact DailyMed mappings.

## 1. Install the project

Confirm Python 3.12 or newer is available:

```powershell
python --version
```

Create the local environment and install TekaRx:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
tekarx --help
```

If PowerShell prevents activation, the environment can be used without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\tekarx.exe --help
```

Do not use `Set-ExecutionPolicy Unrestricted` to solve an activation error. Calling the
executables directly is sufficient.

## 2. Choose the data directory

Commands use `./data` by default. The extractors create only these lifecycle boundaries:

```text
data/
├── raw/          Immutable downloads and manifests
├── interim/      Future normalized Parquet tables
└── processed/    Future model-ready Parquet tables
```

To keep large data on another drive, pass the same directory to every command:

```powershell
tekarx extract-dailymed --dataset rxnorm --data-dir "D:\tekarx-data"
```

Alternatively, set it for the current PowerShell session:

```powershell
$env:TEKARX_DATA_DIR = "D:\tekarx-data"
```

## 3. Download one FAERS quarter

Download the latest quarter listed on the official FDA index:

```powershell
tekarx extract-faers
```

For an initial reproducible experiment, select one complete quarter explicitly:

```powershell
tekarx extract-faers --quarter 2024Q1
```

TekaRx constructs the official FDA ASCII URL for a specified quarter. When the quarter is
omitted, it reads the [FDA FAERS/AEMS quarterly download page][faers-download] and selects
the highest available quarter. Use `--url` only as an advanced override if FDA changes a
specific archive location.

For the lightweight GNN experiment, download the complete temporal split with one command:

```powershell
tekarx extract-faers --preset gnn-small
```

The preset downloads sequentially and safely reuses completed quarters after an interruption:

```text
Train:      2023Q1, 2023Q2, 2023Q3, 2023Q4
Validation: 2024Q1
Test:       2024Q2
```

It writes `data/processed/splits/faers_gnn-small.json`. Later normalization must group all
versions of the same `CASEID` into one split and use `CASEVERSION` for deterministic
deduplication.

Expected files:

```text
data/raw/faers/
├── manifest.json
└── 2024Q1/
    ├── source.zip
    └── extracted/
        ├── .complete
        └── ... official ASCII and README files
```

The quarter must have the form `YYYYQ1`, `YYYYQ2`, `YYYYQ3`, or `YYYYQ4`. The extractor
rejects ZIP entries that attempt to write outside the quarter directory.

## 4. Convert FAERS ASCII to Parquet

After extraction, convert the lightweight split without loading a full source table into RAM:

```powershell
tekarx build-faers --preset gnn-small
```

To convert only one quarter:

```powershell
tekarx build-faers --quarter 2024Q1
```

The default build converts DEMO, DRUG, REAC, OUTC, and DELETE. To select fewer tables:

```powershell
tekarx build-faers --quarter 2024Q1 --tables demo drug reac outc
```

Expected outputs:

```text
data/interim/faers/
|-- manifest.json
|-- demo/2023Q1.parquet ... 2024Q2.parquet
|-- drug/2023Q1.parquet ... 2024Q2.parquet
|-- reac/2023Q1.parquet ... 2024Q2.parquet
|-- outc/2023Q1.parquet ... 2024Q2.parquet
`-- delete/2023Q1.parquet ... 2024Q2.parquet
```

All source fields remain strings during this lossless staging step. Each output adds a
`quarter` column, uses Snappy compression, and embeds the source SHA-256 checksum in Parquet
metadata. A rerun reuses an output only when the current raw TXT checksum matches.
Deletion filenames vary by quarter. When an archive contains multiple deletion-case TXT files
(including the cumulative and incremental files in 2019Q1), the builder streams all numeric
case IDs into the quarter's DELETE Parquet and records every source path and checksum in its
provenance metadata.

## 5. Build the report cohort and graph edges

After all selected quarters are staged, create one latest report per case:

```powershell
tekarx build-cohort
```

Use the full temporal experiment after staging the historical preset:

```powershell
tekarx extract-faers --preset gnn-full
tekarx build-faers --preset gnn-full
tekarx build-cohort --split-preset gnn-full
```

The `gnn-full` split is fixed at 2019Q1–2023Q4 train, 2024Q1 validation, and 2024Q2
test. `gnn-small` uses 2023Q1–2023Q4 for train with the same holdouts.

The root convenience script runs the same command:

```powershell
python build_cohort.py
```

The build order is deliberate:

1. combine quarterly DEMO tables and remove every `caseid` listed by DELETE;
2. rank versions within `caseid` and keep the highest numeric `caseversion`;
3. convert `YR`, `MON`, `WK`, `DY`, `HR`, and `DEC` ages to years, retain missing ages with
   `age_missing = 1`, and remove only known ages outside 0–120;
4. aggregate DRUG, REAC, and OUTC separately by `primaryid`;
5. inner-join drugs, left-join outcomes and reactions, then audit uniqueness;
6. assign the latest version of every `caseid` to an exact source-quarter split;
7. retain source-level report-to-drug, report-to-reaction, and report-to-outcome edges.

Expected outputs:

```text
data/processed/
|-- tekarx_cohort.parquet
|-- case_splits.parquet
|-- cohort_manifest.json
`-- edges/
    |-- report_drug.parquet
    |-- report_reaction.parquet
    `-- report_outcome.parquet
```

`case_splits.parquet` is keyed by `caseid`; this is the leakage boundary. The cohort contains
only the latest `primaryid`, while all retained graph edges carry the same split label as their
report. Reports without a non-empty drug are excluded. A report is positive when OUTC contains
any official serious category (`DE`, `LT`, `HO`, `DS`, `RI`, `CA`, or `OT`); absence of a
documented serious outcome is encoded as zero. Outcome subtypes remain labels or auxiliary
targets and must never be predictors.

DuckDB defaults to a 4 GB memory limit and spills larger operations under `data/interim`. For
a lower-memory machine:

```powershell
tekarx build-cohort --memory-limit 2GB --threads 2
```

After all source/reference staging is complete, the convenient end-to-end command is:

```powershell
tekarx build-prospective --split-preset gnn-small --threads 4
```

It runs cohort → DrugCentral dictionary → prospective feature enrichment → graph/XGBoost in
dependency order. Use `--split-preset gnn-full` only after staging the historical preset.

## 6. Build the drug dictionary

After the cohort and DrugCentral staging tables exist, run:

```powershell
tekarx build-drug-dictionary
```

Or use the equivalent root script:

```powershell
python build_drug_dictionary.py
```

The output is `data/processed/drug_dictionary.parquet` with exactly:

```text
faers_raw, dc_id, atc_code, ror, has_boxed_warning
```

Optionally build the RxNorm bridge first:

```powershell
tekarx build-rxnorm-lookup
tekarx build-drug-dictionary
```

The lookup command scans the DrugCentral dump for either a dedicated RxNorm table or RxNorm
records in `id_type`/`identifier`, then joins local RxCUIs to the staged DailyMed RxNorm names.
It writes `data/interim/drugcentral/rxnorm_lookup.parquet` with:

```text
rxnorm_name, struct_id, rxcui, source, query_name
```

If the local snapshot contains no usable RxNorm identifiers, first build the ordinary drug
dictionary to identify unknown names, then opt into the resumable NLM API pass:

```powershell
tekarx build-drug-dictionary
tekarx build-rxnorm-lookup --use-api
tekarx build-drug-dictionary
```

Use `--max-names 100` for a smoke test. API responses are checkpointed after batches under
`data/interim/rxnorm/rxnav_api_cache.jsonl`; rerunning skips cached queries. The default is 10
requests/second and the command rejects values over NLM's 20 requests/second limit. At roughly
60,000 uncached names, allow at least 100 minutes at the default rate, excluding retries.

The matching policy is deliberately conservative:

1. normalize FAERS and DrugCentral names to uppercase;
2. remove dosage/form terms and common salts such as hydrochloride, sodium, potassium, sulfate,
   maleate, and fumarate;
3. exact-match `prod_ai` against canonical structures and then synonyms;
4. exact-match raw `drugname` when `prod_ai` does not resolve;
5. try exact aliases from `rxnorm_lookup.parquet` for names still unresolved;
6. split explicit `/` and `+` ingredient combinations and preserve every resolved component;
7. enable fuzzy matching only if the unique-name exact hit rate is below 50%, using a default
   score cutoff of 97;
8. fuzzy-match `prod_ai`, then raw `drugname`;
9. reject conflicting mappings and assign `dc_id = 0` to all remaining names.

`drug_dictionary.parquet` is a linkage table: `(faers_raw, dc_id)` is unique, but `faers_raw`
may repeat when an explicit combination resolves to multiple ingredients. Join it to report-drug
edges with this intentional one-to-many behavior.

### Build the graph and baseline

Install the optional ML dependencies and build all artifacts:

```powershell
python -m pip install -e ".[graph]"
tekarx build-graph
```

The command above intentionally builds the base cohort. To rebuild the final graph from the
feature-rescue artifact, pass it explicitly:

```powershell
tekarx build-graph --cohort-path data/processed/tekarx_cohort_feature_rescue.parquet
```

For the tested Windows RTX 4050 setup, install the official CUDA 12.8 wheel if the first command
resolved CPU-only Torch:

```powershell
python -m pip install --force-reinstall "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Consult the official PyTorch installation selector before changing Python, Torch, or CUDA versions.

The graph contains patient and drug node types plus `takes` and reverse `taken_by` relations.
PyG edge indices are contiguous; `drug.node_id` preserves DrugCentral IDs, generated IDs for the
500 frequent unknown names, and semantic ID `-1` for OTHER. Patient features are `age/120`, male
sex as `1` (all other values `0`), and clipped `num_drugs/50`. Drug features are standardized
log train-only ROR, boxed-warning status, and a 26-column ATC-initial multi-hot vector.

The default full-size build streams ordered DuckDB record batches into NumPy `.npy` memory maps
and writes a lightweight graph descriptor. Use `--graph-storage legacy` only for small
compatibility fixtures. The design, RAM-complexity calculation, tuning knobs, and reproducible
capacity estimator are explained in [graph_memory.md](graph_memory.md).

The memory-mapped graph stores one compact patient split ID from `case_splits.parquet`; legacy PyG
mode materializes `train_mask`, `val_mask`, and `test_mask`. The tabular NPZ retains boolean masks
for baseline compatibility. The builder trains XGBoost only on streamed train batches, prints
temporal validation AUC, and saves `xgboost_baseline.json`. ROR is recalculated from train patients
only; the all-split ROR in `drug_dictionary.parquet` is never used as a graph feature.

The single canonical memory-mapped edge list is ordered into train, validation, and test offsets.
The forward patient-to-drug relation is logically restricted to its train interval; consumers swap
the same two index arrays for the all-exposure reverse drug-to-patient relation. Legacy mode
materializes equivalent relations and mutually exclusive edge masks. Unknown-drug vocabulary is
selected from training frequency only; unseen held-out names map to `OTHER`.

Train the graph model after rebuilding the graph:

```powershell
tekarx train-gnn --device cuda
```

Training uses bounded CPU chunks for neighbor aggregation and GPU minibatches for the classifier.
Validation controls early stopping. Test labels are untouched unless `--evaluate-test` is given
after model selection. The saved artifacts are `tekarx_inductive_gnn.pt` and
`tekarx_inductive_gnn_manifest.json`.

### Add patient-level tabular features

Run the enrichment and automatically rebuild the graph and baseline:

```powershell
tekarx add-tabular-features
```

The command writes `data/processed/tekarx_cohort_enriched.parquet`, retaining every original
cohort column and adding:

```text
max_ror, high_ror_count, has_boxed_warning, num_high_risk_atc,
atc_diversity, therapeutic_duplicates,
age_group_0_17, age_group_18_40, age_group_41_64, age_group_65_plus,
num_drugs_squared, polypharmacy_age
```

`num_high_risk_atc` counts distinct drug ingredients in ATC first-level classes N, B, C, or M.
`therapeutic_duplicates` sums `frequency - 1` across each patient's ATC first-level classes.
DrugCentral combination mappings are counted by distinct ingredient ID. All ROR aggregates are
strict-prior-date encodings for training reports; validation and test use one frozen full-training
lookup. Demographic imputation statistics are fitted on training patients only. The enrichment
also adds normalized weight/missingness, route/dose/form/frequency coverage, ATC hierarchy
diversity, and ATC first-level counts.

#### Dosage normalization and leakage controls

The dose normalizer treats a number as meaningful only together with its physical dimension.
It converts compatible units to a canonical scale, while keeping these dimensions separate:

- absolute mass (milligrams);
- mass per body weight (milligrams per kilogram);
- mass per body surface area (milligrams per square metre);
- volume (millilitres);
- international units and international units per kilogram;
- milliequivalents and amount of substance;
- radioactivity (megabecquerels).

An interpretable frequency such as once daily, twice daily, every 12 hours, or weekly can be
converted to administrations per day. Free text, ranges, `PRN`, and other ambiguous schedules do
not produce a daily dose. The pipeline also refuses unsafe cross-dimension conversions: millilitres
are never changed into milligrams without a concentration, international units are never treated
as mass, and percentages or dosage-form counts are not guessed into physical amounts.

Reference statistics are calculated from train rows only and frozen before validation/test are
transformed. A validation/test outlier therefore cannot change its own reference median, spread,
or high-dose threshold. An ingredient-and-dimension reference is used when it has at least 20
training exposures and nonzero log-IQR; otherwise the transform falls back to a training-only
reference for that physical dimension. The fallback never crosses dimensions.

One FAERS `drugname` can map to several DrugCentral ingredients. Because its recorded dose usually
describes the whole combination product, the pipeline does not apply that amount to every mapped
ingredient. Ingredient-relative dose features remain unavailable for ambiguous multi-ingredient
mappings; regimen-level documentation, route, and frequency coverage can still be used.

These features describe what was recorded in the spontaneous report. They do not confirm that a
dose was dispensed or administered, and they should not be interpreted as dosing advice.

The build preserves row-level provenance in
`data/processed/edges/report_drug_dose.parquet` and saves the frozen reference statistics in
`data/processed/dose_normalization_lookup.parquet`. The enriched patient table receives coverage,
relative log amount, relative daily amount, above-training-p90, parenteral-count/fraction, and
IV/SC/IM indicators under the reviewed `dose_normalized_` prefix. The enriched-cohort manifest
records the method, minimum support, clipping bound, combination policy, artifact checksums, and
exact patient-feature list.

#### Test whether normalized dosage helps

After `add-tabular-features` (or `build-prospective`) rebuilds the tabular graph artifact, run the
matched validation-only XGBoost ablation:

```powershell
tekarx evaluate-dosage-ablation --threads 4
```

This trains two models with the same rows, hyperparameters, seed, and temporal masks. The control
removes the reviewed `dose_normalized_` package: relative amount/daily-dose, availability,
scheduling, high-dose, and parenteral features. The older coarse route, formulation,
dose-documentation, and frequency-documentation fields remain in both arms. It reports the paired
validation AUC difference and a stratified paired-bootstrap 95% confidence interval, and writes:

```text
data/processed/dosage_ablation_manifest.json
data/processed/dosage_ablation_validation_predictions.npz
data/processed/xgboost_with_normalized_dosage.json
data/processed/xgboost_without_normalized_dosage.json
```

The command never reads or scores test labels. Treat dosage as a demonstrated improvement when
the validation delta is positive, practically useful, and stable across seeds; a bootstrap
interval entirely above zero is stronger evidence than a point estimate alone. If the interval
crosses zero, keep the result as inconclusive and do not justify a costly full-history rebuild from
that experiment alone.

For a like-for-like GNN sensitivity check, train both tracks with the same seed and settings:

```powershell
tekarx train-gnn --device cuda --feature-track prospective
tekarx train-gnn --device cuda --feature-track prospective-no-dosage
```

Do not add `--evaluate-test` while selecting features. The no-dosage track removes the reviewed
incremental dosage package (relative amount/daily-dose, availability, scheduling, high-dose, and
parenteral features) but leaves the existing coarse exposure fields and the rest of the prospective
graph feature set unchanged.

### Run the feature-rescue package

FAERS stores indications in the separate INDI table, not DRUG. Stage it and run:

```powershell
tekarx build-faers --preset gnn-small --tables indi
tekarx feature-rescue
```

The command fits pair PRR and the indication vocabulary only on train reports whose report year
is at most 2023. It saves:

```text
data/processed/indication_lookup.parquet
data/processed/high_risk_drug_pairs.parquet
data/processed/tekarx_cohort_feature_rescue.parquet
data/processed/tekarx_graph.pt
data/processed/tekarx_tabular_baseline.npz
data/processed/xgboost_baseline.json
```

The pair lookup requires at least 25 train reports per pair by default and retains the 50 largest
finite PRRs for inspection. Model features use strict-prior-date, smoothed pair log-ROR for train
reports and a frozen full-training lookup for later reports. They include maximum/mean pair risk,
scored-pair count, and high-risk-pair count. Indications add three interpretable flags plus 32
deterministic bins fitted from the training indication vocabulary. Reporter occupation is not a
prospective predictor. `is_death` and `is_hospitalization` are attached to patient nodes as
auxiliary targets only; using them to predict `is_serious` would be direct target leakage.

The current `gnn-small` local source covers 2023Q1–2024Q2, not the full 2019–2023 history. The
feature-rescue manifest records the actual training date range so a run cannot silently claim
coverage it does not have.

ATC codes are pipe-separated when a DrugCentral structure has multiple codes. ROR uses distinct
`(primaryid, dc_id)` exposures and the cohort's `is_serious` target, so raw-name aliases mapped
to the same DrugCentral ingredient receive the same ROR. For cells `a`, `b`, `c`, and `d`, it
computes `(a*d)/(b*c)`; only zero-cell tables receive the 0.5 Haldane-Anscombe correction.

The compact DailyMed mapping ZIPs do not contain warning sections. The builder therefore
extracts DrugCentral's DailyMed-derived `active_ingredient`, `prd2label`, and `section` COPY
tables and identifies the FDA SPL boxed-warning LOINC code `34066-1`. It writes the cached
intermediate `data/interim/drugcentral/boxed_warning.parquet` and records its source dump
checksum. Do not interpret the flag as a patient-level diagnosis or causal signal.

## 7. Download DrugCentral

The default command downloads the complete official PostgreSQL dump currently registered
in the extractor:

```powershell
tekarx extract-drugcentral
```

Expected files:

```text
data/raw/drugcentral/
├── drugcentral.dump.11012023.sql.gz
└── manifest.json
```

Check the official [DrugCentral download page][drugcentral-download] before a new project
release. If a newer dump is published, preserve the official URL explicitly:

```powershell
tekarx extract-drugcentral --url "<new-official-dump-url>"
```

This step intentionally retains the original PostgreSQL dump. Build its mapping-relevant
tables without installing PostgreSQL:

```powershell
tekarx build-drugcentral
```

The command scans the gzip stream once and writes:

```text
data/interim/drugcentral/
|-- manifest.json
|-- structures.parquet
|-- synonyms.parquet
|-- atc.parquet
|-- struct2atc.parquet
`-- drug_class.parquet
```

Every database value remains a string at this lossless staging boundary. To build only the
tables needed for basic FAERS name and ATC mapping:

```powershell
tekarx build-drugcentral --tables structures synonyms atc struct2atc
```

## 8. Download compact DailyMed mappings

Download all three small reference packages with one command:

```powershell
tekarx extract-dailymed
```

To download only one mapping, use `--dataset rxnorm`, `--dataset pharmacologic-class`, or
`--dataset metadata`.

Expected files:

```text
data/raw/dailymed/
├── manifest.json
├── rxnorm_mappings.zip
├── pharmacologic_class_mappings.zip
└── dm_spl_zip_files_meta_data.zip
```

Do not download the full multi-part DailyMed label release yet. First use these mappings to
identify labels for drugs present in FAERS.

Convert all downloaded mappings to Parquet:

```powershell
tekarx build-dailymed
```

Or build a single mapping archive:

```powershell
tekarx build-dailymed --dataset rxnorm
```

Expected outputs:

```text
data/interim/dailymed/
|-- manifest.json
|-- rxnorm.parquet
|-- pharmacologic-class.parquet
`-- metadata.parquet
```

Both reference-data builders use bounded batches, Snappy compression, atomic temporary files,
source SHA-256 metadata, and checksum-aware cached reruns.

## 9. Verify provenance and checksums

Every manifest entry records:

- source and dataset name;
- exact download URL;
- UTC retrieval timestamp;
- local path and byte size;
- SHA-256 checksum.

Inspect a manifest:

```powershell
Get-Content data\raw\faers\manifest.json
```

Verify a downloaded file independently:

```powershell
Get-FileHash data\raw\faers\2024Q1\source.zip -Algorithm SHA256
```

If the publisher supplies a checksum, enforce it during extraction:

```powershell
tekarx extract-faers `
  --quarter 2024Q1 `
  --sha256 "<expected-sha256>"
```

## 10. Restart behavior

It is safe to rerun the same command. TekaRx hashes the existing file and reuses it only
when its URL, dataset, path, and manifest checksum agree. The JSON result then contains:

```json
{
  "cached": true
}
```

TekaRx refuses to overwrite an existing file that is missing from its manifest. Investigate
that file before removing it; this prevents an unrelated or manually downloaded artifact
from being silently replaced.

Network failures and HTTP 429 or temporary server errors are retried with bounded backoff.
Incomplete downloads use a `.part` suffix and are never promoted to the final filename.

## 11. Run repository checks

Before committing extraction or transformation changes:

```powershell
ruff check .
pytest
```

The tests use synthetic local content and do not download medical datasets.

## Troubleshooting

### `tekarx` is not recognized

Install the editable package again or call the executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\tekarx.exe --help
```

### Checksum mismatch

Do not disable checksum verification. Confirm that the copied URL and published checksum
refer to the same release. A changed upstream file should be stored as a new, versioned
artifact rather than replacing the existing raw file.

### Existing unverified file

Compare the file with the manifest and source website. Move it to a quarantine directory if
its provenance cannot be established, then rerun extraction. Never edit a raw file in place.

### Disk space

Use `--data-dir` to place datasets on a larger drive. Keep the repository and Python package
on the normal project drive; only the data lifecycle directory needs to move.

[faers-download]: https://www.fda.gov/drugs/fda-adverse-event-monitoring-system-aems/fda-adverse-event-monitoring-system-aems-latest-quarterly-data-files
[drugcentral-download]: https://drugcentral.org/download
