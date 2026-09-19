# TekaRx Capstone System Design and GitHub Ticket Plan

> **Status:** Draft — planning reference only  
> **Last updated:** August 27, 2026  
> **Implementation status:** Not started  
> **Decision gate:** Review and approve this draft before creating GitHub issues, scaffolding the web application, or changing deployment infrastructure.

## 1. Product summary

TekaRx is planned as a synthetic, research-only medication-review prototype built around the frozen GNN produced by the existing offline research pipeline.

```text
Frozen GNN + synthetic cases
          |
          v
Precomputed, versioned prediction bundle
          |
          v
FastAPI -> PostgreSQL/Supabase
          |
          v
React clinician dashboard
```

The capstone application will serve precomputed predictions produced by the real frozen GNN for curated synthetic cases. It will not host the complete training graph or require a production GPU.

### Planned responsibility split

- **Matthew (`matthew-sudo2`) — 65 points:** architecture, model integration, backend, security, evaluation, deployment, and final integration.
- **Clarence — 20 points:** frontend implementation, responsive design, accessibility, and UI integration.
- **Brent — 15 points:** synthetic data, evidence verification, QA, documentation, and presentation support.

Points represent relative effort rather than hours.

## 2. Product and safety boundaries

### Included in the capstone

- A public landing page with a research-use disclaimer.
- Demo authentication using synthetic users.
- A synthetic Patient Notebooks directory with search, filters, and pagination.
- Patient summaries with synthetic demographics and conditions.
- Normalized medication, dose, unit, route, and frequency information.
- A versioned **Model Priority Score** with a frozen threshold.
- Medication mapping coverage.
- GNN feature and medication-contribution explanations.
- Structured boxed-warning, FAERS statistical-signal, and mapping-gap alerts.
- Evidence citations with publisher, URL, version, and supported claim.
- A review workflow with notes, reasons, and immutable audit history.
- Model-card, evaluation, and data-provenance views.

### Explicitly excluded from the capstone

- Real patients, PHI, or real medical-record numbers.
- Live EHR or FHIR integration.
- Live GNN training or GPU inference.
- Claims that a model score is a diagnosis, causal estimate, or verified probability of medication harm.
- Presenting pair PRR as a proven clinical drug-drug interaction.
- Autonomous treatment or medication recommendations.
- GraphRAG, evidence agents, or generated clinical guidance.

### Required interface terminology

- Use **Model Priority Score**, not **Risk Score**.
- Use **No Priority Signal**, not **Stable**.
- Use **Medication Mapping Coverage**, not an unsupported **Model Confidence** value.
- Use **Contribution to Model Score**, not **SHAP value**, for GNN explanations.
- Keep model-priority status separate from workflow states such as New, In Review, Follow-up, and Completed.
- Remove eGFR from contributor examples unless a future validated model actually uses it.
- Replace MRNs with synthetic **Demo IDs**.

Every application screen must identify the data as synthetic and state that TekaRx is research decision support, not a substitute for professional judgment.

## 3. Proposed system architecture

### Offline model and bundle pipeline

1. Freeze the verified full-GNN checkpoint, graph checkpoint, feature schema, and lookup versions.
2. Select an operating threshold using validation data only.
3. Run the locked test set once after all model and threshold decisions are frozen.
4. Assemble six to ten synthetic patient cases using supported inputs.
5. Produce model-ready features using the same frozen transformations used during training.
6. Generate predictions with the actual frozen GNN.
7. Generate feature-occlusion and leave-one-medication-out explanations.
8. Export a versioned bundle containing predictions, explanations, alerts, evidence references, provenance, and hashes.
9. Seed the verified bundle into PostgreSQL for the demo application.

### Planned online components

- **Frontend:** React, TypeScript, Vite, React Router, TanStack Query, Tailwind CSS, and accessible component primitives.
- **Backend:** FastAPI, Pydantic, SQLAlchemy, and Alembic.
- **Database:** PostgreSQL hosted by Supabase for the capstone deployment.
- **Authentication:** Supabase Auth with seeded demo-reviewer accounts.
- **Deployment:** Cloudflare Pages for the frontend and Render for the API.
- **Evidence:** structured, human-reviewed evidence records only.

The online service will not load the complete 8.2-million-node graph. It will serve the frozen demo bundle and review-workflow data.

### Planned API surface

- `GET /api/v1/patients`
- `GET /api/v1/patients/{patient_id}`
- `GET /api/v1/patients/{patient_id}/prediction`
- `GET /api/v1/patients/{patient_id}/alerts`
- `GET /api/v1/evidence/{evidence_id}`
- `GET /api/v1/patients/{patient_id}/reviews`
- `POST /api/v1/patients/{patient_id}/reviews`
- `GET /api/v1/model-info`
- `GET /api/v1/health`

### Planned core entities

- `Patient`
- `MedicationExposure`
- `Prediction`
- `Contribution`
- `Alert`
- `EvidenceCitation`
- `ClinicalReview`
- `AuditEvent`
- `ModelVersion`

Predictions will be immutable. Review actions will append new events rather than update or replace the original prediction.

## 4. Planned user journey

1. The user opens the landing page and sees the research and synthetic-data disclaimer.
2. The user enters the synthetic demo.
3. The Patient Notebooks page loads synthetic cases from the API.
4. The user searches or filters by medication, priority signal, or workflow state.
5. The patient dashboard displays medications, mapping coverage, score, threshold, and version.
6. The user inspects positive and negative model contributors.
7. The user opens alerts and their structured evidence citations.
8. The user starts a review, records a decision or note, and completes the review.
9. The system records the action in an immutable audit timeline.

## 5. GitHub project-management design

### Project board

Create a GitHub Project named **TekaRx Capstone MVP** with this workflow:

```text
Backlog -> Ready -> In Progress -> In Review -> Done
                           |
                           v
                         Blocked
```

Rules:

- Every issue has one primary assignee.
- Matthew may hold at most two issues in progress.
- Clarence and Brent may each hold one issue in progress.
- An issue cannot enter Ready until its dependencies are complete.
- A linked pull request moves the issue to In Review.
- An issue moves to Done only after its pull request and acceptance criteria are complete.

### Milestones

The provisional kickoff date is August 31, 2026.

| Milestone | Provisional due date |
|---|---:|
| Week 1 — Foundation | September 6, 2026 |
| Week 2 — Model Bundle and Read Experience | September 13, 2026 |
| Week 3 — Explanations and Evidence | September 20, 2026 |
| Week 4 — Review Workflow | September 27, 2026 |
| Week 5 — Hardening and Deployment | October 4, 2026 |
| Week 6 — Capstone Release | October 11, 2026 |

### Labels

Planned labels:

- Type: `type:feature`, `type:chore`, `type:test`, `type:docs`
- Area: `area:architecture`, `area:model`, `area:backend`, `area:frontend`, `area:data`, `area:evidence`, `area:security`, `area:devops`
- Priority: `priority:p0`, `priority:p1`
- Risk: `risk:clinical-safety`, `risk:security`
- Workflow exceptions: `blocked`, `needs-review`
- Temporary ownership: `owner:brent`

Project status belongs in the board, so status labels such as `in-progress` should not be created.

### Collaborator onboarding

- Clarence is already a repository collaborator and can be assigned after his exact account is selected.
- Brent does not yet have a GitHub account.
- Until Brent creates an account and accepts a collaborator invitation, his planned issues remain unassigned with `owner:brent`.
- After he accepts, bulk-assign those issues to him and remove the temporary owner label.

## 6. Draft ticket backlog

`TRX-*` is the stable planning identifier retained in the issue title. GitHub will assign a separate issue number when tickets are created.

### Week 1 — Foundation

| ID | Owner | Points | Draft ticket | Acceptance summary |
|---|---|---:|---|---|
| TRX-001 | Matthew | 5 | Freeze MVP architecture, terminology, and safety boundaries | Architecture, synthetic/no-PHI boundary, model terminology, disclaimer, and deferred scope are approved. |
| TRX-002 | Matthew | 3 | Bootstrap React, FastAPI, PostgreSQL, and CI workspace | Both applications start locally; health, lint, tests, build, and secret exclusions work. |
| TRX-003 | Clarence | 5 | Define design system and accessibility blueprint | Tokens, components, responsive breakpoints, keyboard behavior, and corrected labels cover all wireframes. |
| TRX-004 | Brent | 3 | Specify synthetic-patient and evidence fixture contracts | Six-to-ten-case matrix covers mapping gaps, threshold states, evidence states, and contains no real identifiers. |

### Week 2 — Model Bundle and Read Experience

| ID | Owner | Points | Draft ticket | Blocked by | Acceptance summary |
|---|---|---:|---|---|---|
| TRX-005 | Matthew | 8 | Freeze GNN release, threshold, and locked-test evaluation | TRX-001 | Checkpoint hashes and threshold policy are frozen; required discrimination, classification, calibration, and alert-rate metrics are saved. |
| TRX-006 | Matthew | 5 | Implement PostgreSQL schema, migrations, and seed loader | TRX-001, TRX-004 | Seeds are idempotent, predictions immutable, and audit events append-only. |
| TRX-007 | Matthew | 5 | Implement patient and model-information read APIs | TRX-002, TRX-006 | Search, filters, pagination, patient details, model metadata, and error cases match OpenAPI. |
| TRX-008 | Matthew | 3 | Export and verify versioned demo prediction bundles | TRX-005, TRX-010 | Frozen-model scores and provenance are exported; mismatched hashes are rejected. |
| TRX-009 | Clarence | 5 | Build landing page and Patient Notebooks directory | TRX-002, TRX-003, TRX-007 | API-backed pages support search, filters, pagination, synthetic labels, and all page states. |
| TRX-010 | Brent | 4 | Curate and validate synthetic demonstration patients | TRX-004, TRX-005 | Six to ten coherent cases, medication fields, conditions, edge cases, and mapping report are complete. |

### Week 3 — Explanations and Evidence

| ID | Owner | Points | Draft ticket | Blocked by | Acceptance summary |
|---|---|---:|---|---|---|
| TRX-011 | Matthew | 8 | Generate GNN feature and medication contributions | TRX-005, TRX-008, TRX-010 | Feature and medication occlusion results are reproducible and identify their method. |
| TRX-012 | Matthew | 5 | Implement deterministic alert and evidence service | TRX-006, TRX-007 | Priority, boxed-warning, FAERS-pair-signal, and mapping-gap rules return structured evidence. |
| TRX-013 | Clarence | 5 | Build patient dashboard and explanation experience | TRX-003, TRX-007, TRX-011, TRX-012 | Dashboard renders medications, score, threshold, mapping coverage, contributors, alerts, and evidence. |
| TRX-014 | Brent | 4 | Curate and verify evidence citations and alert claims | TRX-004, TRX-010, TRX-012 | Every evidence claim has source, URL, version/date, supported statement, limitation, and medication linkage. |

### Week 4 — Review Workflow

| ID | Owner | Points | Draft ticket | Blocked by | Acceptance summary |
|---|---|---:|---|---|---|
| TRX-015 | Matthew | 5 | Add demo authentication and role-based API authorization | TRX-002, TRX-006, TRX-007 | Seeded reviewer login works; protected operations reject unauthorized access. |
| TRX-016 | Matthew | 5 | Implement review workflow and append-only audit API | TRX-006, TRX-015 | Review transitions are validated and cannot overwrite the original prediction. |
| TRX-017 | Matthew | 3 | Complete frontend/API integration and contract tests | TRX-009, TRX-013, TRX-015, TRX-016 | Authentication, CORS, API errors, and contracts pass without production mock data. |
| TRX-018 | Clarence | 3 | Build review actions and audit-history interface | TRX-013, TRX-016 | Required reasons, notes, transitions, timestamps, and separate status types render correctly. |
| TRX-019 | Brent | 2 | Validate review workflows and failure scenarios | TRX-014, TRX-016, TRX-018 | Happy paths, invalid transitions, expired auth, missing evidence, and mapping gaps are tested. |

### Week 5 — Hardening and Deployment

| ID | Owner | Points | Draft ticket | Blocked by | Acceptance summary |
|---|---|---:|---|---|---|
| TRX-020 | Matthew | 5 | Deploy frontend, API, database, and monitoring | TRX-015, TRX-017, TRX-019 | Deployment works with health checks, logs, rate limiting, and secure configuration. |
| TRX-021 | Matthew | 4 | Publish model card, evaluation report, and integrity audit | TRX-005, TRX-008, TRX-011, TRX-012, TRX-016 | Metrics match artifacts; model limitations, provenance, and UI claims are audited. |
| TRX-022 | Clarence | 2 | Complete responsive, accessibility, and browser QA | TRX-009, TRX-013, TRX-018 | Keyboard, contrast, responsive layouts, Chrome, and Edge pass without blocking defects. |
| TRX-023 | Brent | 1 | Complete acceptance, provenance, and user documentation | TRX-014, TRX-019, TRX-021, TRX-022 | Acceptance report, source/license appendix, user guide, and known limitations are complete. |

### Week 6 — Release

| ID | Owner | Points | Draft ticket | Blocked by | Acceptance summary |
|---|---|---:|---|---|---|
| TRX-024 | Matthew | 1 | Cut and verify the capstone release candidate | TRX-020 through TRX-023 | Clean deployment completes the workflow, CI is green, and hashes and release tag are recorded. |
| TRX-025 | Brent | 1 | Finalize demonstration script and recorded fallback | TRX-023, TRX-024 | Rehearsed walkthrough, presenter notes, reset instructions, and fallback recording are ready. |

Clarence's final screenshots and UI walkthrough are included in TRX-022. Planned effort totals exactly 65 points for Matthew, 20 for Clarence, and 15 for Brent.

## 7. Draft issue template

```markdown
## Goal
One concrete outcome.

## Owner and estimate
Owner:
Reviewer: Matthew
Estimate:
Milestone:

## Dependencies
Blocked by: #...

## Tasks
- [ ] Implementation task

## Acceptance criteria
- [ ] Observable result
- [ ] Required tests pass
- [ ] Documentation is updated

## Out of scope
- Explicit exclusion

## Safety and provenance
- [ ] Synthetic data only
- [ ] No secrets, PHI, or large data committed
- [ ] No unsupported diagnostic or causal claims
- [ ] Model and evidence sources are versioned
```

## 8. Planned issue and pull-request workflow

1. Move a ticket from Backlog to Ready when dependencies close.
2. Assign the ticket before starting it.
3. Create a branch such as `feat/TRX-013-patient-dashboard`.
4. Move the ticket to In Progress.
5. Open a draft pull request early.
6. Add `Closes #<issue-number>` to the pull-request description.
7. Attach tests, screenshots, or evidence required by the ticket.
8. Move the ticket to In Review.
9. Matthew reviews Clarence and Brent's pull requests.
10. Merge only after CI and acceptance criteria pass.

Planned repository governance files, to be added only after this draft is approved:

- `.github/ISSUE_TEMPLATE/task.yml`
- `.github/pull_request_template.md`
- `.github/CODEOWNERS`
- A `main` branch ruleset requiring a pull request, one approval, and passing CI.

## 9. Planned team routine

- **Monday:** select Ready tickets and confirm dependencies.
- **During work:** record blockers on the issue instead of relying on private chat.
- **Friday:** demonstrate completed tickets and move accepted work to Done.
- Requirement changes become new issues or explicit changes to an existing ticket.
- P0 clinical-safety, data-integrity, or security problems block the milestone and release.

## 10. Draft acceptance and release gates

- All displayed people and records are explicitly synthetic.
- The frontend contains no embedded production patient records or model scores.
- Every prediction records model, graph, feature-schema, mapping, and threshold versions.
- Every explanation identifies its attribution method.
- Every medical alert has a structured source or is explicitly marked as a model/statistical signal.
- No interface copy claims diagnosis, causality, verified safety, or calibrated probability without evidence.
- Unauthorized users cannot mutate review data.
- Every review mutation creates an immutable audit event.
- Backend tests, `ruff check .`, frontend tests, accessibility checks, and production builds pass.
- A clean deployment can complete the patient-directory-to-review workflow without manual database changes.

## 11. Items requiring approval before implementation

- Confirm the kickoff date and milestone dates.
- Confirm Clarence's exact GitHub username.
- Add Brent as a collaborator after he creates an account.
- Confirm the final synthetic patient personas.
- Confirm the exact frozen GNN and graph checkpoint IDs.
- Approve the validation-threshold policy before consuming the locked test set.
- Approve the final evidence sources and disclaimer text.
- Approve this document's transition from **Draft** to **Accepted**.

