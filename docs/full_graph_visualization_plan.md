# TekaRx Full-Graph Visualization Implementation Plan

Status: proposed; no implementation authorized yet  
Scope: visualization pipeline only  
Primary target: the verified `gnn-full` patient-drug graph

## 1. Objective

Build an interactive viewer in which every patient node and every retained patient-drug exposure
edge from the full TekaRx graph remains discoverable, while keeping browser memory and frame time
bounded.

“Full visualization” will mean:

- all patient and drug nodes are represented in the exported dataset;
- all retained exposure edges are stored and can be retrieved;
- the initial view uses density, clusters, and aggregated edges;
- individual patients and raw edges appear only at an appropriate zoom level or after selection;
- filtering never silently changes the underlying graph population;
- the manifest reports displayed, loaded, and total counts separately.

It will **not** mean drawing every raw edge at full opacity in every frame. Tens of millions of
overlapping lines would be both slow and visually meaningless.

## 2. Current Baseline

The current pipeline already provides a useful foundation:

- source graph arrays are opened with NumPy memory mapping;
- up to 1,000,000 patients can be selected without materializing the complete graph in RAM;
- patient and drug coordinates are written as raw Float32 buffers;
- patient categories use Uint8 buffers;
- edges use packed Int32 pairs;
- the browser fetches binary assets asynchronously;
- WebGL2 uses instanced point rendering;
- static coordinates avoid browser-side force-layout computation.

The current limitations are:

- a hard one-million-patient export limit;
- one monolithic coordinate buffer per node type;
- one monolithic edge buffer;
- no spatial index or level-of-detail hierarchy;
- no frustum-based data loading;
- no GPU picking for arbitrary patient nodes;
- edge rendering is controlled by a global fraction rather than semantic or spatial visibility;
- the current geometric projection is useful for organization but does not encode learned
  similarity.

All actual graph counts must be read from the selected graph checkpoint manifest. Estimates in this
plan must not become hard-coded assumptions.

## 3. Design Principles

1. **Bounded working memory:** export and viewing memory must depend on shard size and visible level
   of detail, not total graph size.
2. **Progressive disclosure:** show density first, clusters second, and individual records last.
3. **Immutable provenance:** every visualization bundle must identify the graph checkpoint, source
   array hashes, split definitions, layout version, and exporter version.
4. **No label leakage:** visual layout and aggregation must not use validation or test outcomes to
   alter train-derived graph features. Outcome coloring may be applied only as display metadata.
5. **Determinism:** identical graph checkpoint, options, and seed must produce identical partitions,
   coordinates, and aggregates.
6. **Clinical restraint:** the viewer is a research exploration tool, not a diagnostic interface.
7. **Graceful degradation:** lower-memory browsers must be able to reduce level of detail without
   producing a different underlying dataset.

## 4. Proposed Architecture

```text
Memory-mapped graph checkpoint
            |
            v
Offline partition and layout exporter
            |
            +-- global manifest + provenance
            +-- node coordinate/metadata shards
            +-- spatial LOD hierarchy
            +-- aggregated edge shards
            +-- raw adjacency shards
            |
            v
Static HTTP server / Colab preview server
            |
            v
Browser streaming scheduler
            |
            +-- Web Worker decoding
            +-- CPU shard cache
            +-- GPU buffer cache
            +-- frustum and LOD selection
            +-- WebGL2 renderer
            +-- GPU picking and drill-down
```

### 4.1 Offline spatial layout

Replace a single global patient coordinate file with deterministic spatial tiles.

Recommended first layout:

- preserve the bipartite separation between patients and drugs;
- place drugs using stable ordering by train-derived degree and ATC class;
- assign each patient to a primary drug or regimen bucket for coarse locality;
- place patients deterministically inside their bucket using a seeded spatial hash;
- keep outcome labels out of coordinate generation;
- store coordinates as little-endian Float32 triples.

This layout is preferred for the first full implementation because it is explainable, streamable,
and inexpensive. UMAP should be evaluated later on an aggregate representation, not run directly
over millions of raw nodes.

### 4.2 Level-of-detail hierarchy

Build a hierarchy with approximately four levels:

| Level | Display unit | Intended use |
|---|---|---|
| L0 | Global density cells and drug super-groups | Initial complete-graph overview |
| L1 | Regimen/drug clusters | Broad interaction patterns |
| L2 | Patient micro-clusters | Regional exploration |
| L3 | Individual patients and raw edges | Close inspection or explicit selection |

Each aggregate cell should store only counts needed for display, such as:

- patient count;
- serious and non-serious counts;
- split counts;
- retained edge count;
- dominant drugs or ATC classes;
- bounding box and child shard identifiers.

Aggregates are display summaries only and must not be reused as model training features.

### 4.3 Binary partitioning

Do not export one file for the complete graph. Write independently fetchable shards.

Proposed bundle layout:

```text
full_graph_visualization/
  index.html
  manifest.json
  provenance.json
  lod/
    l0_nodes.bin
    l0_edges.bin
    l1_index.bin
    l2_index.bin
  nodes/
    patient_00000.coords.bin
    patient_00000.meta.bin
    ...
    drugs.coords.bin
    drugs.meta.json
  edges/
    aggregate_00000.bin
    ...
    adjacency_00000.bin
    ...
```

Initial shard targets should be benchmarked in the 16–64 MiB range. The exporter should permit a
configurable target size and record the actual size in the manifest.

Every manifest array entry should include:

- relative path;
- logical type;
- dtype and byte order;
- shape;
- byte length;
- SHA-256 checksum;
- node or edge index range;
- spatial bounds;
- LOD level;
- source checkpoint identifier.

### 4.4 Edge storage

Use two complementary edge representations:

1. **Aggregated edges:** cluster-to-drug or cluster-to-cluster counts for overview levels.
2. **Raw adjacency shards:** exact patient-to-drug pairs for close zoom and selected neighborhoods.

Raw edges should be partitioned by patient tile or contiguous patient index range. A compact index
must map each tile to its edge shards. This avoids scanning or downloading all edges when the user
selects one region.

The first implementation should prefer moderate independent files over requiring HTTP byte-range
support. Range requests may be added after confirming consistent behavior in local Python servers,
Colab proxies, and the intended deployment host.

### 4.5 Browser streaming scheduler

Add a scheduler that:

- derives visible spatial tiles from the camera frustum and zoom level;
- requests the appropriate LOD shards;
- cancels stale requests after camera movement;
- decodes metadata in a Web Worker;
- maintains separate bounded CPU and GPU least-recently-used caches;
- reports loading, loaded, visible, and total counts;
- prefetches immediate neighboring tiles during idle periods;
- releases GPU buffers when cache limits are exceeded.

Recommended initial budgets:

| Resource | Initial target |
|---|---:|
| Browser CPU working set attributable to graph | 512 MiB or less |
| GPU graph buffers | 512 MiB or less |
| Single network request | 64 MiB or less |
| Initial interactive payload | 50 MiB or less |
| Initial usable view | 10 seconds or less on local HTTP and SSD |

These are acceptance targets, not promises. They must be measured on the actual Windows/RTX 4050
machine and in the Colab preview path.

### 4.6 Rendering

Retain WebGL2 as the compatibility baseline.

- render density cells and patient points with instanced particles;
- use separate draw calls by node type and LOD;
- perform frustum culling before uploading buffers;
- draw aggregated edges at overview levels;
- draw raw edges only for visible L3 tiles or selected neighborhoods;
- limit labels to drugs, selected nodes, and aggregate summaries;
- use GPU color picking for node selection instead of scanning millions of projected points on the
  CPU;
- preserve the white clinical-dashboard theme and accessible color contrast;
- expose a visible quality control that changes LOD and cache budgets, not graph membership.

WebGPU can be investigated as an optional acceleration track after the WebGL2 implementation meets
correctness requirements. The initial release should not require experimental browser features.

## 5. Backend Refactor Plan

### Phase A: Measurement and contracts

1. Record exact node, drug, edge, split, and dtype counts from the verified graph manifest.
2. Benchmark the current one-million export and browser view.
3. Define and version the sharded visualization manifest schema.
4. Add size-estimation and preflight commands.
5. Establish deterministic test fixtures and golden manifests.

Deliverable: design contract and benchmark report; no full export yet.

### Phase B: Full-node sharded exporter

1. Remove the one-million selection cap behind an explicit `--full` mode.
2. Preserve `--patients` for bounded samples.
3. Stream all patient coordinates and categorical metadata into size-bounded shards.
4. Build deterministic regimen/drug spatial buckets.
5. Write checksums and atomic completion markers.
6. Make interrupted exports resumable at shard boundaries.
7. Report progress in patients, edges, bytes, elapsed time, and estimated remaining time.

Deliverable: complete patient-node bundle with no raw edges loaded by the browser.

### Phase C: LOD and aggregate edges

1. Build L0–L3 spatial indices.
2. Generate train-independent display aggregates for each cell.
3. Generate overview edge summaries.
4. Validate that aggregate counts reconcile exactly with child nodes and edges.

Deliverable: complete graph overview that becomes interactive before individual-node data loads.

### Phase D: Raw adjacency streaming

1. Partition raw patient-drug edges by patient tile.
2. Create tile-to-edge-shard lookup tables.
3. Add neighborhood retrieval for a patient, drug, or visible tile.
4. Verify exact reconciliation against source edge arrays.

Deliverable: every source edge is available on demand without a monolithic browser allocation.

### Phase E: Browser LOD renderer

1. Implement frustum-aware shard loading.
2. Add bounded CPU/GPU caches and cancellation.
3. Add Web Worker parsing and validation.
4. Render LOD particles and aggregate edges.
5. Add GPU picking and selected-neighborhood raw edges.
6. Add diagnostics for frame time, loaded bytes, visible nodes, and cache pressure.

Deliverable: interactive full-graph exploration on the target hardware.

### Phase F: Colab and Drive workflow

1. Extend the Colab notebook to resume/checkpoint sharded exports.
2. Copy only missing or checksum-mismatched shards.
3. Avoid millions of tiny Drive files by enforcing moderate shard sizes.
4. Preview through the Colab HTTP proxy.
5. Package downloadable bundles with their full directory structure.

Deliverable: restartable full visualization generation using Colab compute and Google Drive
storage.

## 6. CLI Proposal

The exact interface should be finalized during Phase A. A likely shape is:

```powershell
tekarx visualize-graph `
  --full `
  --layout hierarchical-bipartite `
  --target-shard-size 32MB `
  --resume `
  --output data/processed/full_graph_visualization/index.html
```

Related diagnostic command:

```powershell
tekarx estimate-visualization `
  --full `
  --graph-dir data/processed/graph_checkpoints/<checkpoint>
```

`--full` must be explicit so an accidental command does not start a long export. It should not be
combined silently with `--patients`.

## 7. Correctness and Leakage Controls

The exporter must verify:

- source arrays match the graph manifest dtype and shape;
- every exported patient appears exactly once at L3;
- all patient indices remain within source bounds;
- every raw exported edge references an exported patient and drug;
- source and exported raw edge counts match exactly when no semantic filter is requested;
- aggregate child counts sum exactly to their parent counts;
- split metadata is copied, not recomputed;
- outcome labels never influence coordinates, partitions, drug ordering, or edge retention;
- no test labels are summarized in a report unless test evaluation has been explicitly authorized;
- bundle manifests preserve the graph checkpoint ID and source provenance.

## 8. Test Strategy

### Unit tests

- deterministic spatial hashing and tile assignment;
- shard boundary behavior;
- dtype, byte order, shape, and checksum validation;
- edge partition and remapping correctness;
- aggregate reconciliation;
- cache eviction and request cancellation;
- manifest path traversal rejection;
- interrupted-export resume behavior.

### Integration tests

- tiny graph with exact expected binary buffers;
- synthetic graph with more nodes than one shard;
- synthetic graph with skewed drug degrees and isolated patients;
- full export restart after an intentionally interrupted shard;
- WebGL template load with mocked binary fetches;
- GPU picking returns the correct node IDs;
- LOD transitions preserve displayed totals.

### Scale tests

Run explicit benchmarks at:

- 100,000 patients;
- 1,000,000 patients;
- the complete graph.

For each, capture:

- export wall time and peak RAM;
- temporary and final disk use;
- bundle size and shard count;
- initial load time;
- browser CPU and GPU memory;
- median and 95th-percentile frame time;
- interaction latency;
- time to reveal one selected patient neighborhood.

Scale tests should be opt-in and must not run in the normal unit-test suite.

## 9. Acceptance Criteria

The full implementation is ready only when all of the following hold:

1. Every source patient and retained edge is represented or retrievable.
2. Initial overview does not require downloading every L3 patient or raw edge shard.
3. Browser working memory stays within the agreed target budget during normal navigation.
4. Initial overview becomes usable within the measured target on the target machine.
5. Camera interaction remains responsive at overview and cluster levels.
6. Selecting a patient or drug retrieves its exact raw neighborhood.
7. Aggregate counts reconcile exactly with raw source counts.
8. Interrupted export and Drive copy operations resume without rebuilding verified shards.
9. No validation/test outcome is used to create the spatial layout.
10. Ruff and the complete unit/integration suite pass.
11. A benchmark report records actual full-graph performance and known hardware limitations.

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Raw edge overdraw | Low frame rate and unreadable view | Aggregate edges by default; raw edges only on demand |
| Browser/GPU buffer duplication | Out-of-memory crashes | Sharded uploads and strict LRU budgets |
| Google Drive small-file overhead | Slow checkpointing | Use moderate shard sizes and resumable manifests |
| Colab runtime reset | Lost export progress | Atomic shard completion markers and resume mode |
| Layout mistaken for clinical similarity | Misinterpretation | Visible disclaimer and explicit layout metadata |
| Outcome-driven spatial leakage | Misleading separation | Prohibit labels from layout and partition logic |
| WebGL implementation limits | Device-specific failures | Preflight limits and automatic lower LOD |
| Picking millions of nodes on CPU | Interaction stalls | GPU color picking and tile-local metadata |
| Manifest or shard corruption | Incorrect graph display | Byte-length and SHA-256 validation |

## 11. Recommended Execution Order

Do not begin with raw full-edge rendering. Implement in this order:

1. benchmark and schema contract;
2. full patient-node sharding;
3. global density and aggregate-edge overview;
4. spatial streaming and cache limits;
5. GPU picking;
6. on-demand raw adjacency;
7. Colab/Drive restartability;
8. complete-graph benchmark and tuning.

The first decision gate should occur after Phase C. If the full-node overview is not stable and
useful on the target browser, raw-edge drill-down should not proceed until memory and frame-time
budgets are corrected.

## 12. Estimated Delivery Shape

This work should be treated as a sequence of independently testable changes rather than one large
rewrite:

1. manifest schema and estimator;
2. sharded patient exporter;
3. LOD aggregation;
4. browser streaming/cache layer;
5. GPU picking and raw-edge drill-down;
6. Colab persistence and scale benchmark.

No production timeline should be committed until Phase A measures the exact full graph and current
one-million baseline on the target hardware.
