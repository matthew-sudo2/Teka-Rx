# TekaRx

TekaRx is a research data pipeline for adverse-drug-event modeling. It extracts immutable
source files from FAERS, DrugCentral, and DailyMed before any normalization or feature
engineering occurs.

> Model outputs are research decision-support signals, not diagnoses or clinical advice.

## Repository layout

```text
src/tekarx/extract/   Source-specific, restartable downloaders
src/tekarx/load/      Atomic Parquet output helpers
tests/                Unit tests using local synthetic files
docs/                 Data provenance and license notes
data/raw/             Immutable source downloads (ignored by Git)
data/interim/         Normalized Parquet tables (ignored by Git)
data/processed/       Model-ready Parquet tables (ignored by Git)
```

The extractors write a SHA-256 checksum and provenance record to a manifest beside each
download. Existing files are reused only when their checksum still matches the manifest.

For a complete Windows walkthrough, expected output trees, restart behavior, checksum
verification, and troubleshooting, see the [extraction runbook](docs/runbook.md).

For the full-history Google Colab workflow, use the staged
[`train_full_colab.ipynb`](notebooks/train_full_colab.ipynb) notebook and read the
[Colab run guide](docs/colab.md). Drive is used for durable checkpoints while cohort building,
memory-mapped graph access, and training run from Colab's local `/content` disk.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matthew-sudo2/Teka-Rx/blob/main/notebooks/train_full_colab.ipynb)

## Installation

Python 3.12 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Extraction

Download the latest available FAERS ASCII quarter automatically:

```powershell
tekarx extract-faers
```

Or select a specific quarter:

```powershell
tekarx extract-faers --quarter 2024Q1
```

Download the recommended lightweight GNN temporal split:

```powershell
tekarx extract-faers --preset gnn-small
```

This preset uses `2023Q1–2023Q4` for training, `2024Q1` for validation, and `2024Q2`
for testing. The command is restart-safe and writes the split definition under
`data/processed/splits/`.

For the production experiment, download five complete training years with fixed holdouts:

```powershell
tekarx extract-faers --preset gnn-full
tekarx build-faers --preset gnn-full
tekarx build-cohort --split-preset gnn-full
```

`gnn-full` assigns `2019Q1` through `2023Q4` to train, `2024Q1` to validation, and
`2024Q2` to test. Missing source quarters are recorded in the cohort manifest and reported as
warnings; a build never silently claims unavailable history.

Convert the extracted preset to RAM-bounded Snappy Parquet tables:

```powershell
tekarx build-faers --preset gnn-small
```

Parquet files are written separately by source table and quarter under
`data/interim/faers/`. Identifiers remain strings, and every file embeds its source path,
SHA-256 checksum, quarter, table name, encoding, and build timestamp.

Build the latest-version, leakage-safe report cohort and graph edges:

```powershell
tekarx build-cohort
```

The builder uses bounded-memory DuckDB operations, applies deleted `caseid` values, keeps the
highest `caseversion`, normalizes age units to years, aggregates DRUG/REAC/OUTC independently,
and assigns each case to exact, non-overlapping source-quarter splits. Missing ages are retained
with an `age_missing` flag; only known ages outside 0–120 are removed. A report is serious when
OUTC contains `DE`, `LT`, `HO`, `DS`, `RI`, `CA`, or `OT`; a report with no documented serious
outcome is the negative class. The equivalent convenience command is `python build_cohort.py`.

Build the FAERS-to-DrugCentral dictionary, serious-outcome ROR, ATC codes, and boxed-warning
flags:

```powershell
tekarx build-drug-dictionary
```

The equivalent root script is `python build_drug_dictionary.py`. Mapping prioritizes normalized
`prod_ai` before the raw `drugname`, first with exact matching and then conservative fuzzy
matching when the exact hit rate is below 50%. ROR is calculated over distinct report/DrugCentral
ID exposures, so all raw aliases mapped to one ingredient share its ROR. Unmapped and ambiguous
names receive DrugCentral ID `0`.

Build an optional DrugCentral/RxNorm bridge before rebuilding the dictionary:

```powershell
tekarx build-rxnorm-lookup
tekarx build-drug-dictionary
```

The first command checks both a literal DrugCentral RxNorm table and its generic
`id_type`/`identifier` tables. If the local snapshot has no usable links, resolve the current
unknown FAERS names through the free NLM RxNorm API with a restartable local cache:

```powershell
tekarx build-rxnorm-lookup --use-api
tekarx build-drug-dictionary
```

API lookup is opt-in because tens of thousands of names can take well over an hour at the
default policy-compliant rate. Explicit `/` and `+` combinations are emitted as multiple
`(faers_raw, dc_id)` linkage rows.

Build the PyTorch Geometric graph and XGBoost baseline:

```powershell
python -m pip install -e ".[graph]"
tekarx build-graph
```

For the final enriched model after `feature-rescue`, select its cohort explicitly:

```powershell
tekarx build-graph --cohort-path data/processed/tekarx_cohort_feature_rescue.parquet
```

On the tested Windows RTX 4050 environment, PyPI initially installed CPU-only Torch. Replace it
with the official CUDA wheel before GNN training, then verify `torch.cuda.is_available()`:

```powershell
python -m pip install --force-reinstall "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Use the current command from the [official PyTorch selector](https://docs.pytorch.org/get-started/locally/)
if this pinned wheel is no longer appropriate for your Python/NVIDIA driver.

This writes `tekarx_graph.pt`, `tekarx_tabular_baseline.npz`, and
`xgboost_baseline.json` under `data/processed/`. Drug-node ROR is recalculated using train
patients only so validation and test outcomes cannot leak into graph features.

Graph arrays are streamed to memory-mapped `.npy` sidecars by default, keeping materialization
RAM proportional to the configured batch instead of total patients or edges. See
[the graph-memory guide](docs/graph_memory.md) for the optimization, capacity estimate, tuning,
and legacy equivalence checks.

Add medication-derived patient features and rebuild the graph/baseline in one command:

```powershell
tekarx add-tabular-features
```

This writes `tekarx_cohort_enriched.parquet` and overwrites the graph and baseline artifacts
with prospective regimen features. Target-derived drug statistics are cross-fitted for training
patients using only earlier reports, then frozen from the complete training period for validation
and test. Missing demographics use training-only imputations and explicit missingness indicators.

Structured FAERS dose amounts are also normalized conservatively. Compatible mass and volume
units are converted to canonical scales, while weight-normalized doses, surface-area-normalized
doses, international units, milliequivalents, and amount-of-substance units remain separate.
Only unambiguous frequencies produce a daily amount. Drug-relative reference statistics are fit
on train only and frozen for validation/test. Product-level doses from combination names are not
duplicated across the mapped DrugCentral ingredients. See the dosage section in
`docs/runbook.md` for the supported and intentionally unsupported cases.

Measure the dosage contribution on validation without touching test labels:

```powershell
tekarx evaluate-dosage-ablation --threads 4
```

The command compares matched XGBoost models with and without the reviewed incremental
`dose_normalized_` package and reports a paired-bootstrap confidence interval for the AUC delta.

Run the complete corrected prospective pipeline with one command after staging FAERS and
DrugCentral:

```powershell
tekarx build-prospective --split-preset gnn-small --threads 4
```

Use `gnn-full` after the historical quarters have been staged. This command rebuilds the cohort,
drug dictionary, cross-fitted features, graph, and XGBoost validation baseline in dependency order.

Train the memory-bounded inductive GNN on the rebuilt graph:

```powershell
tekarx train-gnn --device cuda
```

The trainer uses only training labels, uses validation only for early stopping, and does not read
test labels unless `--evaluate-test` is explicitly supplied.

Run the train-frozen feature-rescue package:

```powershell
tekarx build-faers --preset gnn-small --tables indi
tekarx feature-rescue
```

This saves the top-50 train-only pair PRR lookup, a train-vocabulary indication lookup,
`tekarx_cohort_feature_rescue.parquet`, and rebuilt 25-feature graph/baseline artifacts.
Death and hospitalization are stored as auxiliary targets and are never baseline predictors.

Download the official DrugCentral PostgreSQL dump:

```powershell
tekarx extract-drugcentral
tekarx build-drugcentral
```

The build streams the mapping-relevant `structures`, `synonyms`, `atc`, `struct2atc`, and
`drug_class` COPY blocks directly from the compressed SQL dump. It does not require a local
PostgreSQL server.

Download all three compact DailyMed cross-reference packages:

```powershell
tekarx extract-dailymed
tekarx build-dailymed
```

The DailyMed build produces one normalized Parquet file for each RxNorm, pharmacologic-class,
and SPL metadata mapping archive.

Override the default data directory with `--data-dir` on any command. Run `tekarx --help`
for the complete interface.

## Verification

```powershell
ruff check .
pytest
```

See [data_sources.md](docs/data_sources.md) before redistributing source data.
