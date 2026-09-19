# TekaRx Random Forest Capstone Development Plan

> **Status:** Draft for team and faculty review  
> **Last updated:** September 12, 2026  
> **Supersedes:** the model and application-integration portions of `tekarx_capstone_system_and_ticket_plan_draft.md`  
> **Does not replace:** the existing GNN and XGBoost research artifacts, evaluation documents, or data-provenance records.

## 1. Decision record

The capstone application will use a **Random Forest classifier** as its only production-facing model.

The existing models have separate, deliberately limited roles:

| Model | Role in this capstone | May drive the UI or API? | Purpose |
|---|---|---:|---|
| Random Forest | Primary model | Yes | The reproducible, faculty-compliant model used for the demo prediction bundle. |
| XGBoost | Tabular benchmark | No | A matched tabular comparator trained and evaluated under the same frozen split. |
| Existing GNN | Historical research benchmark | No | An archived graph-model comparator; no new GNN work, graph serving, GPU inference, or graph explanation is part of the capstone application. |

This distinction is important: **a higher historical GNN AUROC does not make it the capstone model.** The professor's restriction governs the deliverable. The GNN may be cited in the methods/results appendix only as an offline benchmark, with its limitations and incomplete evaluation disclosed.

Current historical validation evidence, recorded in `docs/gnn_full_evaluation.md`, is GNN AUROC 0.888018 and XGBoost AUROC 0.880653 on the 2024 Q1 selection split. These are not Random Forest results and must not be copied into the Random Forest model card or dashboard.

## 2. Product direction from the frontend already built

The static React prototype is the product baseline, not a screen to rebuild:

```text
Landing page (#home)
        |
        v
Patient Notebooks (#patient-notebooks)
        |
        v
Patient Overview (#patient-overview)
```

The current prototype already provides:

- A public landing page with research-only and synthetic-data language.
- A six-case Patient Notebooks directory.
- A patient overview with medications, alerts, a model panel, and review actions.
- A shared dark forest-green visual system, responsive layouts, and accessibility-aware controls.

The next work is to replace static fixture values with a versioned Random Forest demo bundle. It is **not** permission to add real patient data, EHR integration, or live clinical decision-making.

### Required frontend terminology correction before integration

The present mockup uses terms such as `Risk Score`, `Model Confidence`, `Stable`, and `MRN`. Before any model bundle is connected, the UI must use the following safer terms:

| Current prototype wording | Required wording |
|---|---|
| Risk Score | Model Priority Score |
| Model Confidence | Model version and/or Medication Mapping Coverage |
| Stable | No Priority Signal |
| MRN | Demo ID |
| Known Interaction | Evidence or statistical-signal label that names its source and limitation |

The score represents a model output for **recorded serious FAERS outcomes**, not a diagnosis, causal finding, verified probability of drug harm, or recommendation to change a medication.

## 3. Target architecture

```text
Leakage-safe FAERS feature tables
             |
             v
Random Forest training + validation-only model selection
             |
             +--> XGBoost matched benchmark report
             +--> archived GNN benchmark table (no serving)
             |
             v
Versioned synthetic demo bundle (JSON/Parquet + manifest + hashes)
             |
             v
FastAPI read/review API + PostgreSQL/Supabase
             |
             v
Existing React landing, notebooks, and patient overview
```

For the first integrated demo, precomputed bundle scores are preferred over live inference. This keeps the demo reproducible, avoids exposing the training data, and makes it possible to validate every displayed result against a model hash.

## 4. Model and evaluation contract

### 4.1 Data split and leakage controls

Use the same temporal cohort definition for every comparison:

| Split | Intended period | Use |
|---|---|---|
| Train | 2019 Q1 - 2023 Q4 | Fit transformations and Random Forest candidates. |
| Validation | 2024 Q1 | Select feature set, hyperparameters, calibration method, and operating threshold. |
| Locked test | 2024 Q2 | Evaluate once after every model decision is frozen. |

Rules:

1. A `caseid` must never cross split boundaries; retain only the latest valid report version per case before splitting.
2. Fit imputers, encoders, dosage transforms, ROR/PRR lookups, indication statistics, feature selection, calibration, and threshold selection on training data only.
3. Apply frozen artifacts unchanged to validation and test records.
4. Do not use a GNN-derived feature, embedding, score, or explanation in the Random Forest feature matrix.
5. Preserve source file hashes, cohort manifests, feature schema, random seed, package versions, and split identifiers.

### 4.2 Random Forest primary model

The initial primary estimator is `sklearn.ensemble.RandomForestClassifier` with:

- A version-pinned scikit-learn dependency.
- Stratified or class-weighted training selected on validation data only.
- A small, declared validation search over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, and `class_weight`.
- Fixed `random_state`, `n_jobs`, and training feature order.
- Out-of-bag statistics kept as a training diagnostic only, never as final performance evidence.
- A serialized estimator, ordered feature schema, preprocessing manifest, calibration artifact if used, and score threshold in one immutable model release.

Random Forest is chosen for faculty compliance, transparent feature handling, CPU-friendly reproducibility, and a credible feature-importance story. It is not assumed to be the highest-AUROC model before matched evaluation.

### 4.3 Matched comparisons

Train XGBoost and Random Forest on the **same feature schema, temporal split, label, preprocessing policy, and validation/test records**. Report:

- AUROC and AUPRC.
- Precision, recall/sensitivity, specificity, F1, accuracy, PPV, and NPV at a validation-chosen threshold.
- Calibration curve, Brier score, and calibration slope/intercept when feasible.
- Alert rate and confusion matrix at the declared threshold.
- Paired bootstrap confidence intervals for the AUROC and AUPRC differences.
- Runtime, peak RAM, estimator size, and inference latency for reproducibility.

The historical GNN can appear in a separate benchmark row only. Its published value must be labeled as historical validation evidence, not a fresh head-to-head result, unless all models are reevaluated under the exact same frozen protocol.

### 4.4 Locked-test gate

Before evaluating 2024 Q2, freeze and version:

- Random Forest parameters and random seed.
- Feature schema and preprocessing artifacts.
- Calibration choice, if any.
- Operating threshold and its clinical/operational rationale.
- Metric code and subgroup definitions.
- Synthetic-case bundle generation code.

If any of these change after the test is used, label the test result as historical evidence for the prior release and create a new future holdout for the revised model.

## 5. Team ownership and branch workflow

The local clone currently only has `main` and `origin/main` visible, so the exact remote names of the newly created Clarence and Brent branches should be confirmed in the kickoff issue. This plan refers to them as **Clarence branch** and **Brent branch** without guessing their names.

| Owner | Primary branch responsibility | Review owner | Guardrails |
|---|---|---|---|
| Matthew | Random Forest pipeline, leakage controls, benchmark evaluation, bundle exporter, backend contract, integration, and release | Faculty/team review where required | Owns all model and data decisions; no teammate edits the frozen split or model manifest without review. |
| Clarence | Frontend fidelity, accessible components, responsive behavior, API-state integration, and screenshots | Matthew | Works from the existing React prototype; does not invent model metrics or clinical claims. |
| Brent | Synthetic fixtures, evidence/provenance records, QA scenarios, documentation, and acceptance testing | Matthew | Uses synthetic IDs only; verifies copy against the model card and evidence sources. |

Branch routine:

1. Create one issue per deliverable and assign a single owner.
2. Branch from updated `main`; use `feat/RF-###-short-name` or `docs/RF-###-short-name`.
3. Keep one concern per pull request. Do not merge generated FAERS data, model binaries, Drive exports, secrets, or notebook checkpoints.
4. Open a draft PR early; add screenshots for UI work and manifests/metrics for model work.
5. Matthew reviews Clarence and Brent changes; at least one reviewer approves before merge.
6. Rebase or merge current `main`, run the required checks, then merge the PR and close its issue.

## 6. Phased development plan

### Phase 0 - Reset the model contract

**Owner:** Matthew  
**Output:** an approved, non-GNN capstone model specification.

- [ ] Record the professor restriction in the README, model card, and presentation outline.
- [ ] Add and pin scikit-learn; do not remove existing graph dependencies yet because historical notebooks still need to open.
- [ ] Define a leakage-safe tabular feature allowlist and data dictionary.
- [ ] Write a Random Forest experiment specification: seed, candidate grid, validation metrics, calibration policy, and threshold policy.
- [ ] State that XGBoost is a matched benchmark and GNN is archival/historical only.

**Exit criterion:** the team can answer, in one sentence, “Which model powers the app?” with “the versioned Random Forest release.”

### Phase 1 - Establish reproducible Random Forest evidence

**Owner:** Matthew, with Brent QA support  
**Output:** a reproducible primary-model and benchmark report.

- [ ] Implement `train-random-forest` and `evaluate-tabular-benchmarks` commands or equivalent versioned scripts.
- [ ] Reuse the prospective cohort and feature-rescue tables; do not rebuild the graph.
- [ ] Fit all transformations on train only and verify them with tests.
- [ ] Select the RF candidate and threshold using validation only.
- [ ] Train a matched XGBoost benchmark.
- [ ] Produce a benchmark table with RF, XGBoost, and separately labeled historical GNN evidence.
- [ ] Generate validation predictions, threshold table, calibration artifacts, and paired bootstrap outputs.
- [ ] Freeze the release manifest before a single locked-test evaluation.

**Exit criterion:** all reported RF numbers point to persisted prediction files and a manifest; no metric is copied from GNN/XGBoost output.

### Phase 2 - Build the synthetic demo bundle

**Owners:** Matthew and Brent  
**Output:** six to ten coherent synthetic cases with immutable RF outputs.

- [ ] Brent prepares a synthetic-case matrix: demographics, conditions, medication, normalized dose/route/frequency, mapping coverage, alert state, and evidence state.
- [ ] Matthew validates each case against the frozen RF feature schema and exports predictions.
- [ ] Generate feature contributions using a declared tree-model method (permutation importance globally; a reproducible local method such as TreeSHAP only if approved and version-pinned).
- [ ] Add structured evidence records with source, URL, version/date, supported statement, limitation, and medication linkage.
- [ ] Produce `demo_bundle.json`, a machine-readable manifest, and a human-readable model card. Hash every artifact.

**Exit criterion:** every number and alert in the UI can be traced to a synthetic input, RF release, feature schema, and evidence record.

### Phase 3 - Integrate the existing frontend

**Owners:** Clarence leads; Matthew integrates API/model contract; Brent verifies content  
**Output:** the current landing, notebook directory, and patient overview read from the demo bundle/API.

- [ ] Preserve `#home`, `#patient-notebooks`, and `#patient-overview` during the transition.
- [ ] Replace static patient fixtures with typed bundle/API data while retaining loading, empty, and error states.
- [ ] Replace prototype wording using the terminology table in Section 2.
- [ ] Display RF release ID, threshold policy summary, feature availability/mapping coverage, and synthetic-data disclaimer.
- [ ] Build an evidence drawer/panel that distinguishes label evidence, statistical signal, model contribution, and a missing mapping.
- [ ] Keep review actions clearly separate from predictions; no action may change the original model result.

**Exit criterion:** the UI has no hard-coded model score or false GNN claim, and each synthetic screen remains readable on desktop, tablet, and mobile.

### Phase 4 - Review workflow, QA, and release evidence

**Owners:** Matthew backend/release; Clarence accessibility/UI QA; Brent acceptance/provenance QA  
**Output:** a credible capstone demonstration package.

- [ ] Implement an append-only synthetic review log or a clearly static simulation if backend scope is deferred.
- [ ] Test search, filters, routing, missing evidence, unavailable mapping, invalid workflow transition, and empty/error states.
- [ ] Run keyboard, contrast, responsive, Chrome, and Edge checks.
- [ ] Publish the Random Forest model card, benchmark report, provenance appendix, and known limitations.
- [ ] Make a rehearsal script that explicitly says the app is research-only and the GNN is not the deployed model.

**Exit criterion:** a clean machine or deployment can show the full synthetic workflow without a GPU, GNN package, private data, or manual artifact replacement.

## 7. Draft issue backlog

| ID | Owner | Area | Deliverable | Depends on |
|---|---|---|---|---|
| RF-001 | Matthew | model/governance | Approve Random Forest model contract and retire GNN from serving scope | None |
| RF-002 | Matthew | data/model | Define leakage-safe RF feature schema and preprocessing manifest | RF-001 |
| RF-003 | Matthew | model | Implement deterministic Random Forest training and release serialization | RF-002 |
| RF-004 | Matthew | evaluation | Implement matched RF vs XGBoost evaluation and historical-GNN benchmark table | RF-003 |
| RF-005 | Brent | data/QA | Create synthetic-case and evidence fixture contract | RF-001 |
| RF-006 | Matthew + Brent | model/data | Export and verify versioned RF synthetic demo bundle | RF-003, RF-005 |
| RF-007 | Clarence | frontend | Convert prototype terminology and create typed UI data contract | RF-001 |
| RF-008 | Clarence | frontend | Integrate Patient Notebooks and Overview with bundle/API states | RF-006, RF-007 |
| RF-009 | Brent | evidence/docs | Verify displayed evidence, provenance, and limitation copy | RF-005, RF-006 |
| RF-010 | Matthew | backend/integration | Implement read-only bundle/API adapter and model-info endpoint | RF-006 |
| RF-011 | Clarence | frontend/QA | Accessibility, responsive, and browser QA for all three routes | RF-008 |
| RF-012 | Brent | QA/docs | Acceptance test, demo script, and provenance appendix | RF-009, RF-011 |
| RF-013 | Matthew | release | Final integration, benchmark report, and release gate | RF-004, RF-010, RF-012 |

## 8. Acceptance criteria

The capstone is ready only when all of the following are true:

- The application states that Random Forest is the model powering its synthetic demonstration bundle.
- XGBoost is described only as a matched benchmark; GNN is described only as archival/historical research evidence.
- No GNN weights, embeddings, score, graph topology, or graph explanation is used by the app.
- Random Forest and XGBoost are evaluated on the same frozen temporal cohort, with data-leakage controls documented.
- The locked test split is consumed once, only after model decisions are frozen, or remains explicitly unconsumed.
- Every UI model value has a release ID, feature/schema version, threshold policy, and source artifact.
- Every user-facing person, Demo ID, medication scenario, score, alert, and review is synthetic.
- The UI never calls a model output a diagnosis, causal result, safety guarantee, or medication recommendation.
- Frontend build, frontend checks, backend tests, `ruff check .`, and data/model tests pass.
- The final presentation includes the RF results, matched XGBoost comparison, and a clearly caveated historical GNN benchmark table.

## 9. Immediate next meeting agenda

1. Confirm the exact names of the Clarence and Brent branches and link them to RF-007/RF-008 and RF-005/RF-009/RF-012.
2. Approve the Random Forest feature allowlist and temporal split contract.
3. Decide whether the first demo reads a local signed bundle or a minimal FastAPI endpoint.
4. Assign RF-001, RF-005, and RF-007; these can start in parallel without touching the locked test set.
5. Do not start RF-003, RF-004, or RF-006 until the feature contract is reviewed.
