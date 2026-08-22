# Running `gnn-full` on Google Colab

Use the [`train_full_colab.ipynb`](../notebooks/train_full_colab.ipynb) notebook for the
full-history experiment. It keeps the existing local/Drive `Teka-Rx/data` experiment untouched
and writes the full run under:

```text
/content/drive/MyDrive/Teka-Rx-full/data
```

The fixed experiment is:

- train: `2019Q1` through `2023Q4`;
- validation: `2024Q1`;
- locked test: `2024Q2`.

Do not add later FAERS quarters to this data root. Cohort construction selects the latest version
of each `caseid` across the staged source, so unrelated future versions could change which report
is retained.

## Storage design

Google Drive provides durable capacity; it does not add RAM or local disk to the Colab virtual
machine. The notebook therefore uses this flow:

```text
Drive raw/interim sources
          |
          | verified copy
          v
/content/tekarx-data  -> DuckDB/features -> memory-mapped graph -> CUDA training
          |
          | verified checkpoint
          v
Drive processed artifacts and model
```

Do not run the memory-mapped graph or temporary neighbor aggregation directly through the Drive
mount. Drive has higher latency and operation quotas; `/content` is the fast, ephemeral work
area. Google recommends minimizing mounted-Drive reads and copying active archives/data to the
runtime where practical. See the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html).

The notebook requires at least 35 GiB of free Drive space and recommends an 80 GiB reserve. The
larger figure is operational headroom, not an expected artifact size. A run with 35–80 GiB free
can proceed, but monitor Drive after every checkpoint and keep unrelated experiments outside the
dedicated root. The notebook calculates only the bytes still needing a copy and refuses the local
copy unless `/content` has those bytes plus 45 GiB of build headroom. On restart it removes only
known interrupted DuckDB/bucket scratch under `/content/tekarx-data/interim`; it never removes
durable Drive inputs.

## Before opening Colab

The notebook clones GitHub. Commit and push the reviewed implementation before running it, then
replace the `GIT_REF` placeholder with the full reviewed commit SHA. Mutable refs such as
`origin/main` are rejected. The notebook records the resolved SHA
and package/runtime versions beside the artifacts. Never put a GitHub personal access token,
Google service-account file, or other credential in the notebook.

Current TekaRx metadata requires Python 3.12 or newer. Use Colab's current runtime and its
preinstalled compatible CUDA-enabled Torch rather than installing the Windows RTX 4050 wheel.
Current and pinnable runtime versions are listed in the
[official runtime-version page](https://research.google.com/colaboratory/runtime-version-faq.html).

## After a runtime reset

Assume every Python variable, import, package install, and `/content` artifact is gone. Drive
checkpoints remain. In a CPU build runtime, rerun Mount Drive through Dependency audit, then the
Experiment plan and Source audit cells. The three source-staging cells will print `SKIP` when the
source marker is valid. Continue with Restore build inputs, Build features, Verify features, Build
graph, and Verify graph. Do not advance to CUDA training on a CPU runtime.

After changing to a GPU runtime, rerun only Mount Drive through Dependency audit, then Restore
graph, GPU audit, Train GNN, and Persist model. The restore cell imports its own graph loader and
does not depend on state from the CPU graph-audit cell. If the GPU runtime resets, repeat that GPU
sequence; training resumes from the latest compatible epoch checkpoint in Drive.

## Stage 1: persistent source staging

Use a standard CPU runtime. The notebook:

1. mounts Drive and writes a frozen experiment definition;
2. downloads and builds DrugCentral;
3. downloads all 22 FAERS quarters and builds six Snappy-Parquet tables per quarter;
4. optionally builds DailyMed and the local DrugCentral/RxNorm bridge;
5. verifies all `22 x 6` FAERS table/quarter combinations and exactly one DrugCentral SQL dump;
6. writes `_SOURCES_SUCCESS.json` only after validation passes.

FAERS and DrugCentral downloads are checksum-cached. FAERS table conversion is also cached by
source checksum. A valid `_SOURCES_SUCCESS.json` makes expensive staging cells print `SKIP`; the
source-audit cell still rechecks the Parquet footers before advancing. A killed HTTP request
restarts only the current file. Historical FDA ZIPs with no deletion-case text file receive a
provenance-marked empty `delete` Parquet tied to the source archive.

If a runtime stops after a ZIP was created but before `extracted/.complete`, inspect that exact
quarter. Move only its incomplete `extracted` directory aside before retrying. Do not delete or
replace the complete raw source tree.

## Stage 2: cohort, dosage features, and graph

Choose a high-memory CPU runtime if available and rerun the notebook setup cells. The notebook
copies these prerequisites to `/content/tekarx-data`:

- only the FAERS, DrugCentral, and optional DailyMed interim directories;
- exactly one raw DrugCentral `.sql.gz` dump;
- only the artifacts required by the deepest valid processed-stage checkpoint.

Raw FAERS ZIP and TXT files remain in Drive. The build is split into cohort, dictionary,
tabular/dosage, and feature-rescue commands. Every stage uploads its own files and writes its
success marker last, so a reset resumes from the deepest valid stage. The graph is built
separately with:

```text
storage          memory-mapped
Arrow batch      131,072 rows
XGBoost batch     65,536 rows
DuckDB memory          6 GB
DuckDB threads             1
```

On a standard Colab runtime with about 12.7 GiB of system RAM, the notebook caps DuckDB at
6 GB, uses one worker thread, disables insertion-order preservation, and spills oversized
intermediates to `/content/tekarx-data/interim/.duckdb_temp/`. The remaining memory is reserved
for Python, Arrow, the operating system, and filesystem cache. Do not raise this above 7 GB on
that runtime; use a high-memory runtime if one query still cannot complete with spilling enabled.
The cohort builder additionally projects only the required DEMO columns before version ranking and
hash-partitions drug, reaction, outcome, and wide edge rows into 64 temporary Parquet buckets. This
bounds the otherwise non-spillable ordered string-aggregation and edge-deduplication states while
preserving the final pipe-separated lists and separate graph edge tables. Bucket files are deleted
after the atomic cohort build.

The tabular-feature stage uses the same 64-way patient hash boundary for ingredient mapping,
strict-prior-date ROR attachment, ATC/exposure/dosage aggregation, and the final wide patient join.
Dosage reference fitting is separately sharded by DrugCentral ingredient, computes each exact
quantile vector once per group, and normalizes one edge shard at a time. The stage writes
bucket-local Snappy parts and streams those parts into the atomic final Parquet file. Its temporary
bucket tree is local to `data/interim` and is removed on both success and failure; an interrupted
run never replaces a previous completed feature artifact.

The feature-rescue stage is likewise 64-way bounded. It generates unique ingredient pairs per
patient bucket, repartitions by drug pair, computes strict-prior-date training histories one pair
bucket at a time, and reduces patient partials immediately. Its train-only indication vocabulary
is deduplicated and partitioned before the 32 hash counts are built. This removes the former
global pair/history/ASOF state that exhausted the 6 GB DuckDB limit.

The graph audit requires:

- `tekarx.memmap_graph` format;
- train-only unknown-drug vocabulary and ROR;
- no held-out patient-to-drug messages into shared drug nodes;
- death/hospitalization auxiliary targets excluded from patient features;
- one row per `primaryid` and one retained version per `caseid`;
- no missing quarter in `cohort_manifest.json`.

Graph sidecars are uploaded into a unique `processed/graph_checkpoints/<checkpoint-id>/` staging
directory, size/hash checked, and renamed before `_GRAPH_SUCCESS.json` publishes its pointer. The
previous graph stays available until its replacement model is durable. A runtime must never
restore a graph checkpoint without this marker.

## Stage 3: CUDA training

Switch to a GPU runtime and rerun setup. Restore the versioned graph descriptor and arrays to
`/content`, then train with the prospective feature track.
The temporary neighbor matrix is local and removed after training.

The current trainer supports CPU/CUDA, not TPU. A TPU port would require PyTorch/XLA device,
synchronization, loader, and checkpoint changes; selecting a TPU in Colab is not sufficient. See
the [PyTorch/XLA migration guide](https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html).

Validation controls early stopping. The notebook deliberately contains no command that consumes
test labels. Training writes a small atomic state checkpoint to Drive after epoch 1, every five
epochs, at the final requested epoch, and when early stopping fires. It includes model, optimizer,
best validation state, normalization, history, and RNG state; resume rejects a different graph or
hyperparameters. The final model is published under
`processed/gnn_checkpoints/<checkpoint-id>/`, then `_GNN_SUCCESS.json` is written last.

## Resume table

| Stage | Resume behavior |
|---|---|
| Raw download | Checksum cache; the active partial request restarts |
| FAERS/DrugCentral Parquet | Source-checksum cache |
| Cohort | `_COHORT_SUCCESS.json`; restore exact cohort and edge files |
| Dictionary | `_DICTIONARY_SUCCESS.json`; restore cohort plus dictionary |
| Dosage/enrichment | `_TABULAR_SUCCESS.json`; restore through tabular artifacts |
| Feature rescue | `_RESCUE_SUCCESS.json` / `_FEATURES_SUCCESS.json` |
| Graph | Restore the versioned directory referenced by `_GRAPH_SUCCESS.json` |
| GNN | Resume the compatible atomic Drive checkpoint from its last saved epoch |

Managed Colab hardware and runtime durations are dynamic. Free runtimes may terminate before a
long build completes; use the staged checkpoints and consider a paid high-memory/GPU runtime for
the full run. Buying more Drive space does not increase Colab VM disk or RAM.

## Durable outputs

Keep these together under Drive `processed/`:

```text
cohort and edge Parquet files
drug dictionary and train-frozen lookups
graph_checkpoints/<checkpoint-id>/
gnn_training_checkpoints/<graph-and-run-id>.pt
gnn_checkpoints/<checkpoint-id>/
colab_run_metadata.json
_*_SUCCESS.json
```

## Outcome-label interpretation

FAERS `OUTC_COD` values represent documented serious outcomes. Reports without an OUTC row supply
the negative class in this cohort; they are **not** verified-safe reports. Dropping all missing
outcomes would remove the negative class under the current target definition. Interpret
`is_serious` as “documented serious outcome versus no documented serious outcome,” retain the
split-level missing-outcome audit, and do not make causal or clinical-safety claims from it.

Model outputs are research signals for retrospective evaluation. They do not demonstrate that a
drug caused an event and must not be used as diagnoses or treatment recommendations.
