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

Allow about 80 GiB of Drive headroom. The notebook also calculates the exact size of the staged
inputs and refuses the local copy unless `/content` has that space plus 30 GiB of build headroom.

## Before opening Colab

The notebook clones GitHub. Commit and push the reviewed implementation before running it, then
replace `GIT_REF = "origin/main"` with the full commit SHA. The notebook records the resolved SHA
and package/runtime versions beside the artifacts. Never put a GitHub personal access token,
Google service-account file, or other credential in the notebook.

Current TekaRx metadata requires Python 3.12 or newer. Use Colab's current runtime and its
preinstalled compatible CUDA-enabled Torch rather than installing the Windows RTX 4050 wheel.
Current and pinnable runtime versions are listed in the
[official runtime-version page](https://research.google.com/colaboratory/runtime-version-faq.html).

## Stage 1: persistent source staging

Use a standard CPU runtime. The notebook:

1. mounts Drive and writes a frozen experiment definition;
2. downloads and builds DrugCentral;
3. downloads all 22 FAERS quarters and builds six Snappy-Parquet tables per quarter;
4. optionally builds DailyMed and the local DrugCentral/RxNorm bridge;
5. verifies all `22 x 6` FAERS table/quarter combinations and exactly one DrugCentral SQL dump;
6. writes `_SOURCES_SUCCESS.json` only after validation passes.

FAERS and DrugCentral downloads are checksum-cached. FAERS table conversion is also cached by
source checksum. Rerunning a completed cell is safe. A killed HTTP request restarts only the
current file; completed quarters remain cached.

If a runtime stops after a ZIP was created but before `extracted/.complete`, inspect that exact
quarter. Move only its incomplete `extracted` directory aside before retrying. Do not delete or
replace the complete raw source tree.

## Stage 2: cohort, dosage features, and graph

Choose a high-memory CPU runtime if available and rerun the notebook setup cells. The notebook
copies these prerequisites to `/content/tekarx-data`:

- all interim FAERS, DrugCentral, and optional DailyMed Parquet files;
- exactly one raw DrugCentral `.sql.gz` dump;
- a prior processed checkpoint when its completion marker exists.

Raw FAERS ZIP and TXT files remain in Drive. The prospective pipeline is run with `--skip-graph`,
then its cohort and features are validated and checkpointed. The graph is built separately with:

```text
storage          memory-mapped
Arrow batch      131,072 rows
XGBoost batch     65,536 rows
DuckDB memory          4 GB
```

The graph audit requires:

- `tekarx.memmap_graph` format;
- train-only unknown-drug vocabulary and ROR;
- no held-out patient-to-drug messages into shared drug nodes;
- death/hospitalization auxiliary targets excluded from patient features;
- one row per `primaryid` and one retained version per `caseid`;
- no missing quarter in `cohort_manifest.json`.

Graph sidecars are uploaded before the descriptor and manifests. `_GRAPH_SUCCESS.json` is written
last. A runtime must never restore a graph checkpoint without this marker.

## Stage 3: CUDA training

Switch to a GPU runtime and rerun setup. Restore the descriptor and the complete
`tekarx_graph_arrays/` directory to `/content`, then train with the prospective feature track.
The temporary neighbor matrix is local and removed after training.

The current trainer supports CPU/CUDA, not TPU. A TPU port would require PyTorch/XLA device,
synchronization, loader, and checkpoint changes; selecting a TPU in Colab is not sufficient. See
the [PyTorch/XLA migration guide](https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html).

Validation controls early stopping. The notebook deliberately contains no command that consumes
test labels. After training, it asserts that `test_auc` is null and `test_evaluated` is false,
then uploads the model and writes `_GNN_SUCCESS.json` last.

The GNN currently saves atomically after training but has no epoch-resume checkpoint. A runtime
termination during training requires restarting that training cell. Source, feature, and graph
checkpoints remain reusable.

## Resume table

| Stage | Resume behavior |
|---|---|
| Raw download | Checksum cache; the active partial request restarts |
| FAERS/DrugCentral Parquet | Source-checksum cache |
| Cohort/dictionary | Input/build-version cache where supported |
| Dosage/enrichment/rescue | Atomic files; rerun the active build cell |
| Graph | Restore only with `_GRAPH_SUCCESS.json` |
| GNN | No partial-epoch resume; restore graph and retrain |

Managed Colab hardware and runtime durations are dynamic. Free runtimes may terminate before a
long build completes; use the staged checkpoints and consider a paid high-memory/GPU runtime for
the full run. Buying more Drive space does not increase Colab VM disk or RAM.

## Durable outputs

Keep these together under Drive `processed/`:

```text
cohort and edge Parquet files
drug dictionary and train-frozen lookups
tekarx_graph.pt
tekarx_graph_arrays/
graph_manifest.json
tekarx_inductive_gnn.pt
tekarx_inductive_gnn_manifest.json
colab_run_metadata.json
colab_training_metadata.json
_*_SUCCESS.json
```

Model outputs are research signals for retrospective evaluation. They do not demonstrate that a
drug caused an event and must not be used as diagnoses or treatment recommendations.
