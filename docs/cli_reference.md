# TekaRx CLI reference

## Overview

The TekaRx CLI downloads immutable FAERS, DrugCentral, and DailyMed source data, converts it
to reproducible Parquet tables, builds the leakage-aware cohort and drug graph, enriches
tabular features, trains the inductive GNN, and exports graph visualizations. Commands use
`data/` by default; pass the same `--data-dir` to every command when data lives elsewhere.

Install the project with the optional dependencies needed by the stage you plan to run:

```powershell
python -m pip install -e ".[graph]"
python -m pip install -e ".[imrad]"
```

Verify the installation:

```powershell
tekarx --help
```

The parser currently exposes the commands below. There is no `download-faers`,
`build-graph-features`, or `evaluate-gnn` command in `tekarx.cli`; use `extract-faers`,
`build-graph`, and `train-gnn --evaluate-test` respectively.

## Command reference

### `tekarx extract-faers`

**Purpose:** Download one FAERS quarterly ASCII ZIP, or all quarters in a named experiment preset.

**Inputs:** FDA FAERS download page or `--url`; optional checksum.

**Outputs:** `data/raw/faers/<quarter>/source.zip`, extracted source files, and
`data/raw/faers/manifest.json`; preset runs also write `data/processed/splits/faers-<preset>.json`.

**Example:**

```powershell
tekarx extract-faers --quarter 2024Q1
```

**Flags:** `--data-dir data` (or `TEKARX_DATA_DIR`), `--sha256` expected checksum,
`--quarter` such as `2024Q1`, `--preset` one of the supported temporal presets, and `--url`
an advanced FDA ZIP URL override. `--quarter` and `--preset` are mutually exclusive.

### `tekarx extract-drugcentral`

**Purpose:** Download the DrugCentral PostgreSQL dump used for name, structure, and ATC linkage.

**Inputs:** The default DrugCentral URL or `--url`.

**Outputs:** `data/raw/drugcentral/` archive and `manifest.json`.

**Example:**

```powershell
tekarx extract-drugcentral --data-dir data
```

**Flags:** `--data-dir data`, `--sha256` expected checksum, and `--url` (default is the
official DrugCentral dump URL).

### `tekarx extract-dailymed`

**Purpose:** Download compact DailyMed mapping archives.

**Inputs:** The selected official DailyMed dataset URL.

**Outputs:** `data/raw/dailymed/` archive files and `manifest.json`.

**Example:**

```powershell
tekarx extract-dailymed --dataset rxnorm
```

**Flags:** `--data-dir data`, `--sha256` expected checksum, `--dataset` (omit to download all
supported datasets), and `--url` for a single-dataset URL override.

### `tekarx build-faers`

**Purpose:** Convert extracted FAERS ASCII tables to lossless Snappy Parquet.

**Inputs:** `data/raw/faers/<quarter>/extracted/` source files.

**Outputs:** `data/interim/faers/<table>/<quarter>.parquet` and
`data/interim/faers/manifest.json`.

**Example:**

```powershell
tekarx build-faers --preset gnn-small
```

**Flags:** `--data-dir data`; exactly one of required `--quarter` or `--preset`; and
`--tables demo drug reac outc delete` (default: all core tables and deletion tables).

### `tekarx build-drugcentral`

**Purpose:** Convert the DrugCentral dump into mapping-relevant Parquet tables.

**Inputs:** `data/raw/drugcentral/` extracted dump.

**Outputs:** `data/interim/drugcentral/*.parquet` and `data/interim/drugcentral/manifest.json`.

**Example:**

```powershell
tekarx build-drugcentral
```

**Flags:** `--data-dir data` and `--tables` (default: all mapping-relevant tables; valid table
choices are reported by `tekarx build-drugcentral --help`).

### `tekarx build-dailymed`

**Purpose:** Convert downloaded DailyMed compact mappings to Parquet.

**Inputs:** `data/raw/dailymed/` archives.

**Outputs:** `data/interim/dailymed/*.parquet` and `data/interim/dailymed/manifest.json`.

**Example:**

```powershell
tekarx build-dailymed --dataset rxnorm
```

**Flags:** `--data-dir data` and `--dataset` (omit to build all downloaded datasets).

### `tekarx build-cohort`

**Purpose:** Deduplicate FAERS cases, assign temporal splits, and build the report-level cohort
and report-to-drug, report-to-reaction, and report-to-outcome edges.

**Inputs:** `data/interim/faers/` Parquet tables and the selected split preset.

**Outputs:** `data/processed/tekarx_cohort.parquet`, `case_splits.parquet`, `edges/*.parquet`,
and `cohort_manifest.json`.

**Example:**

```powershell
tekarx build-cohort --split-preset gnn-full --memory-limit 8GB --threads 4
```

**Flags:** `--data-dir data`; `--split-preset gnn-small` (default); `--memory-limit 4GB`
(default); and optional `--threads`.

### `tekarx build-drug-dictionary`

**Purpose:** Link FAERS raw names to DrugCentral IDs and compute train-frozen drug ROR values.

**Inputs:** The processed cohort, report-drug edges, and DrugCentral interim mappings.

**Outputs:** `data/processed/drug_dictionary.parquet`, its
`drug_dictionary_manifest.json`, and supporting boxed-warning artifacts.

**Example:**

```powershell
tekarx build-drug-dictionary --fuzzy-score-cutoff 97
```

**Flags:** `--data-dir data`; `--fuzzy-trigger-rate 0.50`; `--fuzzy-score-cutoff 97.0`;
`--fuzzy-margin 3.0`; `--memory-limit 4GB`; and optional `--threads`.

### `tekarx build-rxnorm-lookup`

**Purpose:** Build the local DrugCentral-to-RxNorm name bridge and optionally query RxNav for
remaining names.

**Inputs:** DrugCentral and DailyMed interim mappings; optionally the RxNav API.

**Outputs:** `data/interim/drugcentral/rxnorm_lookup.parquet`,
`rxnorm_lookup_manifest.json`, and resumable API cache files when `--use-api` is enabled.

**Example:**

```powershell
tekarx build-rxnorm-lookup --use-api --max-names 100
```

**Flags:** `--data-dir data`; `--use-api` disabled by default; `--batch-size 250`;
`--requests-per-second 10.0`; and optional `--max-names`.

### `tekarx build-graph`

**Purpose:** Materialize the patient-drug graph and train the XGBoost tabular baseline.

**Inputs:** `data/processed/tekarx_cohort.parquet` or `--cohort-path`, the dictionary, and
report-drug edges.

**Outputs:** `tekarx_graph.pt` or mmap graph arrays, `graph_manifest.json`, graph descriptors,
and XGBoost baseline artifacts.

**Example:**

```powershell
tekarx build-graph --cohort-path data/processed/tekarx_cohort_feature_rescue.parquet
```

**Flags:** `--data-dir data`; optional `--cohort-path`; `--top-unknown 500`; `--memory-limit 4GB`;
optional `--threads`; `--xgb-rounds 500`; `--xgb-early-stopping 30`; `--xgb-max-depth 5`;
`--xgb-max-leaves 0`; `--xgb-device cpu` (`auto` or `cuda` are also valid);
`--graph-storage memory-mapped` (`legacy` is also valid); `--materialization-batch-size 131072`;
and `--xgb-batch-size 65536`.

### `tekarx evaluate-dosage-ablation`

**Purpose:** Compare validation XGBoost AUC with and without normalized dosage features.

**Inputs:** The enriched tabular artifact, or `--tabular-path`, and the processed split data.

**Outputs:** Validation predictions and `dosage_ablation_manifest.json`.

**Example:**

```powershell
tekarx evaluate-dosage-ablation --xgb-rounds 1000 --bootstrap-samples 500
```

**Flags:** `--data-dir data`; optional `--tabular-path`; `--xgb-rounds 1000`; 
`--xgb-early-stopping 50`; `--xgb-max-depth 0`; `--xgb-max-leaves 63`; optional `--threads`;
`--seed 42`; and `--bootstrap-samples 500`.

### `tekarx add-tabular-features`

**Purpose:** Add prospective dosage and exposure features and optionally rebuild graph artifacts.

**Inputs:** The cohort, dictionary, report-drug edges, and dosage lookup inputs.

**Outputs:** `tekarx_cohort_enriched.parquet`, `cohort_enriched_manifest.json`, dosage lookup
artifacts, and graph artifacts unless `--skip-graph` is used.

**Example:**

```powershell
tekarx add-tabular-features --memory-limit 8GB --threads 4
```

**Flags:** `--data-dir data`; `--memory-limit 4GB`; optional `--threads`; and `--skip-graph`
(disabled by default).

### `tekarx feature-rescue`

**Purpose:** Fit train-frozen indication and high-risk pair features, then retrain downstream
artifacts.

**Inputs:** The enriched cohort, drug dictionary, and report edges.

**Outputs:** `tekarx_cohort_feature_rescue.parquet`, `feature_rescue_manifest.json`, and graph
artifacts unless `--skip-graph` is used.

**Example:**

```powershell
tekarx feature-rescue --training-end-year 2023 --top-pairs 50
```

**Flags:** `--data-dir data`; `--training-end-year 2023`; `--top-pairs 50`;
`--minimum-pair-reports 25`; `--memory-limit 4GB`; optional `--threads`; and `--skip-graph`.

### `tekarx build-prospective`

**Purpose:** Run cohort, dictionary, feature, and graph construction in dependency order.

**Inputs:** Staged FAERS, DrugCentral, and DailyMed data.

**Outputs:** All artifacts from the four pipeline stages, including manifests and graph files.

**Example:**

```powershell
tekarx build-prospective --split-preset gnn-full --threads 4
```

**Flags:** `--data-dir data`; `--split-preset gnn-small`; `--memory-limit 4GB`; optional
`--threads`; `--fuzzy-trigger-rate 0.50`; `--fuzzy-score-cutoff 97.0`; `--fuzzy-margin 3.0`;
and `--skip-graph`.

### `tekarx train-gnn`

**Purpose:** Train the leakage-safe inductive patient-drug GNN.

**Inputs:** Graph artifact, graph array manifest, cohort labels, and optional checkpoint.

**Outputs:** `tekarx_inductive_gnn.pt` or the selected feature-track model,
`tekarx_inductive_gnn_manifest.json`, checkpoints, and optional test predictions.

**Example:**

```powershell
tekarx train-gnn --epochs 100 --batch-size 8192 --device cpu --evaluate-test
```

**Flags:** `--data-dir data`; optional `--graph-path`; `--epochs 100`; `--batch-size 8192`;
`--hidden-channels 64`; `--dropout 0.20`; `--learning-rate 0.001`; `--weight-decay 0.0001`;
`--patience 15`; optional `--device`; `--seed 42`; `--edge-chunk-size 250000`;
optional `--checkpoint-path` and `--resume-from`; `--checkpoint-every 5`;
`--feature-track prospective`; and `--evaluate-test` disabled by default.

### `tekarx estimate-visualization`

**Purpose:** Estimate disk and memory requirements for a full graph visualization export.

**Inputs:** Graph checkpoint, array directory, or manifest.

**Outputs:** A JSON estimate printed to stdout; no graph data is modified.

**Example:**

```powershell
tekarx estimate-visualization --graph-dir data/processed/tekarx_graph_arrays
```

**Flags:** `--data-dir data`; optional `--graph-dir`; and `--target-shard-size 32MB`.

### `tekarx visualize-graph`

**Purpose:** Export a sampled interactive graph, or a complete resumable sharded visualization.

**Inputs:** Graph arrays/checkpoint and the selected patient/drug split.

**Outputs:** HTML/JSON visualization files and an output manifest; `--full` writes sharded
assets and can resume with `--resume`.

**Example:**

```powershell
tekarx visualize-graph --split validation --layout 3d --patients 100 --top-drugs 50
```

**Flags:** `--data-dir data`; optional `--graph-dir`; `--split validation` (`train` or `test`);
`--layout 3d` (`2d` or `hierarchical-bipartite`); `--patients 100`; `--top-drugs 50`;
`--full`; `--target-shard-size 32MB`; `--resume`; `--seed 42`; and optional `--output`.

## End-to-end workflow

This sequence starts in a clean repository with Python 3.12 or newer. The `gnn-full` preset
matches the documented 2019Q1-2023Q4 train, 2024Q1 validation, and 2024Q2 test split.

```powershell
python -m pip install -e ".[dev,graph,imrad]"

# Step 1 - Download raw data
tekarx extract-faers --preset gnn-full
tekarx extract-drugcentral
tekarx extract-dailymed

# Step 2 - Convert source data to Parquet
tekarx build-faers --preset gnn-full
tekarx build-drugcentral
tekarx build-dailymed

# Step 3 - Build the cohort and drug dictionary
tekarx build-cohort --split-preset gnn-full
tekarx build-rxnorm-lookup
tekarx build-drug-dictionary

# Step 4 - Add prospective features and build graph artifacts
tekarx add-tabular-features
tekarx build-graph

# Step 5 - Train and evaluate the inductive GNN
tekarx train-gnn --feature-track prospective --evaluate-test

# Step 6 - Optional graph visualization
tekarx visualize-graph --split validation --layout 3d
```

For a single command after all source staging is complete, `tekarx build-prospective --split-preset
gnn-full` runs the cohort, dictionary, feature, and graph stages in order.

## Troubleshooting

**`tekarx` is not recognized.** Activate the virtual environment, or call
`.venv\Scripts\tekarx.exe` directly after `python -m pip install -e ".[dev]"`.

**CUDA out of memory.** Use `--device cpu`, reduce `--batch-size` and `--edge-chunk-size`, or
lower `--hidden-channels`. The GNN can resume from `--resume-from` after an interrupted run.

**RxNorm rate limit or timeout.** Leave `--use-api` off when local mappings are sufficient. If
the API is needed, keep `--requests-per-second` at or below 20, use `--max-names` for a smoke
test, and rerun; cached responses are reused.

**Missing manifest.** Run the preceding extraction/build stage with the same `--data-dir`.
For graph commands, point `--graph-dir` at `data/processed/tekarx_graph_arrays` or its
`manifest.json`.

**Out-of-memory during cohort or feature construction.** Lower `--memory-limit` and
`--threads`; DuckDB will spill intermediate work under `data/interim`.

**A command says an input table is missing.** Confirm the raw quarter was extracted, the
corresponding `build-faers` command completed, and that the current PowerShell session is using
the same `TEKARX_DATA_DIR`.

## Manifest files

The following are the stable manifest families written by the CLI. Generic `manifest.json`
files are scoped by their containing lifecycle directory.

| Manifest | Description |
|---|---|
| `data/raw/faers/manifest.json` | FAERS archive URLs, checksums, and extraction records. |
| `data/raw/drugcentral/manifest.json` | DrugCentral source archive provenance. |
| `data/raw/dailymed/manifest.json` | DailyMed archive provenance. |
| `data/interim/faers/manifest.json` | FAERS ASCII-to-Parquet conversion records. |
| `data/interim/drugcentral/manifest.json` | DrugCentral table conversion records. |
| `data/interim/drugcentral/boxed_warning_manifest.json` | DailyMed boxed-warning linkage provenance. |
| `data/interim/dailymed/manifest.json` | DailyMed mapping conversion records. |
| `data/interim/drugcentral/rxnorm_lookup_manifest.json` | RxNorm lookup inputs, cache, and output details. |
| `data/processed/cohort_manifest.json` | Cohort, split, deletion, and edge provenance. |
| `data/processed/drug_dictionary_manifest.json` | Name matching, ROR, and dictionary provenance. |
| `data/processed/cohort_enriched_manifest.json` | Prospective dosage/exposure feature provenance. |
| `data/processed/feature_rescue_manifest.json` | Train-frozen rescue feature provenance. |
| `data/processed/graph_manifest.json` | Graph storage, node/edge, and baseline artifact metadata. |
| `data/processed/tekarx_graph_arrays/manifest.json` | Memory-mapped graph array descriptors. |
| `data/processed/dosage_ablation_manifest.json` | Dosage ablation configuration and validation results. |
| `data/processed/tekarx_inductive_gnn_manifest.json` | Prospective GNN model and evaluation provenance. |
| `data/processed/tekarx_no_dosage_inductive_gnn_manifest.json` | No-dosage GNN comparison provenance. |
| `data/processed/full_graph_visualization/index/manifest.json` | Full visualization shard index and export metadata. |
| `data/processed/graph_1m_assets/manifest.json` | Sampled visualization asset metadata. |