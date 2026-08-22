# Memory-bounded full graph build

The full-history graph is larger than the six-quarter development graph, but its arrays do not
need to be resident in RAM at the same time. TekaRx therefore uses a memory-mapped graph store by
default and keeps the legacy in-memory PyG build as a small-data compatibility mode.

## Where the old peak came from

The legacy builder collected complete DuckDB results into Arrow/Python/NumPy objects before it
wrote the graph. At different points it could hold:

- a complete patient table and Python lists for its columns;
- the full `N x Fp` patient feature matrix;
- separate query, stacked, reversed, and train-filtered edge-index arrays;
- split arrays, masks, labels, PyTorch tensors, and serialization buffers;
- additional XGBoost training and validation copies.

Some NumPy-to-Torch conversions share storage, but column selection, boolean filtering, index
reversal, standardization, and model-matrix construction allocate new buffers. Chunking only the
final GPU minibatches cannot recover RAM already consumed during graph materialization.

## The memory-mapped design

The default graph build follows this data path:

```text
DuckDB ordered query
    -> bounded Arrow record batch
    -> vectorized conversion into one slice
    -> open NumPy .npy memory map
    -> lightweight tekarx_graph.pt descriptor + versioned array manifest
```

Patient features, labels, split IDs, drug features, and canonical patient/drug edge indices live
in `data/processed/tekarx_graph_arrays/`. The `.pt` file is a descriptor, not a second copy of the
large tensors. Arrays are written once in deterministic index order. The builder uses 32-bit edge
indices whenever the node cardinalities fit; this halves edge-index storage relative to `int64`.

One canonical edge list stores every patient-drug exposure in train/validation/test order. The
manifest stores the edge split offsets, while each patient's split is stored once as `int8`. The train-only
patient-to-drug topology is a logical view of edges whose target patient is in train; the reverse
drug-to-patient view retains all splits. This preserves the inductive rule without materializing a
second reversed array. Held-out patients still cannot send messages into shared drug nodes.

During materialization, private working memory is proportional to the configured batch, not the
whole cohort:

```text
disk             O(N * Fp + D * Fd + E)
builder work RAM O(B * Fp + B * edge_columns + D * Fd)
```

Here `N` is patients, `D` drugs, `E` unique exposure edges, `Fp`/`Fd` feature counts, and `B` the
materialization batch size. Memory-mapped file size is disk usage, not automatically private RAM;
Windows may show recently touched mapped pages in the process working set and can evict clean
pages when another process needs memory.

## Capacity estimate for `gnn-full`

Run the transparent planning model before downloading or rebuilding the complete history:

```powershell
python benchmarks/graph_memory_estimate.py
```

Its defaults model 8 million patients, 4,000 drug nodes, 24 million edges, 118 patient features,
28 drug features, 32-bit indices, and a 131,072-row materialization batch. The current estimate is:

```text
memory-mapped sidecars on disk       3.79 GiB
pre-optimization legacy minimum      4.34 GiB
memory-mapped batch working set      0.12 GiB
working-array reduction              37.6x
```

The legacy number is intentionally a lower bound. It excludes Python strings/lists, Arrow validity
buffers, allocator overhead, XGBoost, serialization, and OS page cache; observed legacy peak RSS
can be substantially higher. The mapped number is the graph-materialization batch working set,
not a promise for end-to-end process RSS. The saved array manifest records actual shapes, dtypes,
and bytes so estimates can be replaced with measured artifact sizes after a build.

Supply different assumptions explicitly:

```powershell
python benchmarks/graph_memory_estimate.py `
  --patients 8000000 `
  --edges 24000000 `
  --patient-features 118 `
  --materialization-batch-size 65536
```

## Build commands and tuning

Use the memory-mapped mode for the full build:

```powershell
tekarx build-graph `
  --cohort-path data/processed/tekarx_cohort_feature_rescue.parquet `
  --graph-storage memory-mapped `
  --materialization-batch-size 131072 `
  --xgb-batch-size 65536 `
  --xgb-device auto `
  --memory-limit 4GB `
  --threads 4
```

`--memory-limit` controls DuckDB. `--materialization-batch-size` controls the largest Arrow-to-NPY
slice and is the first setting to reduce if graph materialization is tight on RAM. Try 65,536 and
then 32,768 rows. `--xgb-batch-size` independently bounds the dense batches handed to XGBoost's
`QuantileDMatrix`; XGBoost's quantized matrix and training workspace are not included in the graph
materialization estimate. Smaller batches reduce transient input memory roughly linearly at the
cost of more batch overhead. Keep the arrays on a fast local SSD and allow enough free space for
temporary and final artifacts during an atomic rebuild.

`--xgb-device cuda` accelerates only histogram boosting and its evaluation. The DuckDB joins,
Parquet scans, ordering, and NPY materialization that construct the graph remain CPU/SSD work.
`--xgb-device auto` selects CUDA only when both PyTorch can see a GPU and the installed XGBoost
wheel reports CUDA support. The command prints the resolved device before expensive graph work.

For a tiny fixture or compatibility comparison, use:

```powershell
tekarx build-graph --graph-storage legacy
```

Legacy mode intentionally materializes a complete PyG `HeteroData` object and is not recommended
for `gnn-full` on a 24 GB laptop.

Train the descriptor-backed graph locally on CUDA:

```powershell
tekarx train-gnn `
  --device cuda `
  --batch-size 8192 `
  --edge-chunk-size 250000
```

The trainer opens sidecars read-only, calculates normalization statistics from train rows in
chunks, and normalizes only gathered minibatches. It does not create a second full standardized
patient matrix. Neighbor aggregation consumes the canonical edge arrays in chunks and writes its
`N x Fd` result to a temporary memory map, which is removed when training finishes. For the default
8-million-patient/28-drug-feature estimate, that temporary file is about 0.83 GiB. The only
cohort-length private working vector is the float32 neighbor-degree vector (about 31 MiB at 8
million patients); feature-shaped working memory remains chunk- or minibatch-bounded:

```text
temporary trainer disk O(N * Fd)
extra trainer RAM      O(N + C * (Fp + Fd) + model)
GPU RAM                O(training_batch * (Fp + Fd + hidden_width) + model)
```

`C` is `--edge-chunk-size`, which is also used for bounded scans and training-only moment
calculation. Reducing it bounds temporary edge/index and statistics buffers; reducing
`--batch-size` primarily lowers GPU and gathered-feature memory. Test labels remain unevaluated
unless `--evaluate-test` is given after every modeling choice is frozen. The GNN manifest records
the storage backend, temporary-neighbor backend, chunk size, minibatch normalization policy, and
leakage controls.

## How to verify the optimization

Run the equivalence and architecture guards:

```powershell
python -m pytest tests/test_graph_memory.py tests/test_inductive_gnn.py -q
python -m ruff check .
```

The tests compare memory-mapped arrays with a legacy graph on the same fixture, assert that split
and train-only edge semantics are unchanged, verify mmap-backed loading, and guard against
reintroducing full-table collection and stacked/reversed edge copies. Numerical trainer tests
compare chunked statistics/aggregation with eager reference calculations and verify that changing
validation or test values cannot alter train-fitted normalization statistics.

For a real build, also inspect `data/processed/graph_manifest.json` and
`data/processed/tekarx_graph_arrays/manifest.json`. Record Windows Task Manager's peak committed
memory alongside the manifest when reporting an experiment; working-set numbers alone can include
reclaimable mapped pages.
