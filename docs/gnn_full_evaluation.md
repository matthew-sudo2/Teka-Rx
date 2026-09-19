# TekaRx `gnn-full` evaluation report and evidence-based evaluation framework

**Report date:** 24 August 2026  
**Model checkpoint:** `gnn-d974b6ad582a-20260823T140402976481Z`  
**Graph checkpoint:** `graph-d974b6ad582a-20260823T134817274791Z`  
**Intended use:** research decision-support and pharmacovigilance triage; not diagnosis, causal inference, or autonomous clinical decision-making

## 1. Executive assessment

The current `gnn-full` result is promising but not yet a complete model evaluation.

The verified model achieved a **temporal validation AUROC of 0.8880** on 2024 Q1. The corresponding full-data XGBoost baseline achieved **0.8807**, an absolute difference of **+0.0074 AUROC** in favor of the GNN. This supports the conclusion that the graph model has useful discriminative ability and that the graph representation may add signal beyond the tabular baseline.

That conclusion must remain narrow. The saved artifact does not contain validation probabilities, so the following have not yet been measured for `gnn-full`: AUPRC, confidence intervals, precision, recall, F1, specificity, NPV, calibration, alert reduction, decision-curve net benefit, subgroup performance, or explanation fidelity. The test quarter remains locked. Therefore:

- **Supported now:** the selected model ranks documented-serious reports above other reports reasonably well in one future quarter.
- **Not supported now:** that its probabilities are accurate, that any alert threshold is clinically safe or useful, that it reduces workload without unacceptable misses, that it generalizes beyond FAERS, or that the model identifies causal drug harm.
- **Readiness decision:** suitable for continued retrospective research; not ready for clinical or regulatory deployment.

## 2. Evidence audited

### 2.1 Artifact integrity and provenance

The local copies of all three files recorded in `data/processed/_GNN_SUCCESS.json` were checked against their recorded byte sizes and SHA-256 hashes:

| Artifact | Size check | SHA-256 check |
| --- | ---: | ---: |
| `colab_training_metadata.json` | Pass | Pass |
| `tekarx_inductive_gnn.pt` | Pass | Pass |
| `tekarx_inductive_gnn_manifest.json` | Pass | Pass |

The success marker records:

- Git revision: `d974b6ad582a9731f9968d70e50fbddc090c8d1d`
- Split preset: `gnn-full`
- GPU: Tesla T4
- PyTorch: 2.11.0 with CUDA 12.8
- Seed: 42
- Test evaluated: `false`

The evaluation in this report uses the immutable model manifest and graph manifest. It does not substitute metrics from the smaller local `gnn-small` graph.

### 2.2 Cohort and temporal design

| Split role | Source quarters | Patients | Documented serious | Other/no documented serious | Positive prevalence |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | 2019 Q1–2023 Q4 | 7,410,073 | 4,327,337 | 3,082,736 | 58.40% |
| Validation/model selection | 2024 Q1 | 369,838 | 200,551 | 169,287 | 54.23% |
| Locked test | 2024 Q2 | 397,113 | Not inspected for this report | Not inspected for this report | Not inspected |

The cohort builder retains only the latest `caseversion` for each `caseid` before assigning the report to a split. This prevents multiple versions of the same case from crossing train, validation, and test.

The validation set is temporal, which is stronger than a random holdout for estimating near-future performance. It is still a **model-selection set**, because its AUROC controlled early stopping. It must not be presented as the final unbiased performance estimate. The untouched 2024 Q2 test set is the single-use final internal temporal evaluation set.

### 2.3 Graph and model specification

| Component | `gnn-full` specification |
| --- | --- |
| Patient nodes | 8,177,024 |
| Drug nodes | 4,000 |
| Patient–drug exposure edges | 23,606,304 |
| Patient features | 118 |
| Drug features | 28 |
| Architecture | One-hop mean drug-neighbor aggregation plus patient self-features |
| Hidden width | 64 |
| Dropout | 0.20 |
| Trainable parameters | 18,049 |
| Batch size | 8,192 patients |
| Edge scan chunk | 250,000 edges |
| Maximum epochs | 100 |
| Early-stopping patience | 15 epochs |
| Epochs completed | 58 |
| Best epoch | 43 |

The graph manifest documents the following leakage controls:

- the unknown-drug vocabulary was selected from training patients only;
- drug ROR and normalization statistics were fitted using training data only;
- optimization used training labels only;
- early stopping used validation labels only;
- test labels were not evaluated;
- held-out patient-to-drug messages were disabled, preventing validation or test patients from updating shared drug representations.

## 3. Current performance assessment

### 3.1 Discrimination

| Model | Selection data | AUROC | 95% CI | Status |
| --- | --- | ---: | ---: | --- |
| TekaRx `gnn-full` | 2024 Q1 | **0.888018** | Not calculated | Current selected model |
| Full-data XGBoost baseline | 2024 Q1 | 0.880653 | Not calculated | Comparator |
| Absolute GNN difference | Same reports | **+0.007365** | Not calculated | Statistical superiority not established |

The comparison is paired because the models use the same validation reports. A paired patient-level bootstrap should be used to estimate the confidence interval for the AUROC difference. Until that interval is available, the observed improvement should be described as numerical rather than statistically established.

#### Traditional metric comparison: current availability

Traditional classification metrics require two items that AUROC does not provide: each report's
predicted probability and a declared decision threshold. Neither model's full-data validation
probabilities were retained in the immutable graph/model checkpoint. Consequently, these values
cannot be reconstructed from the two AUROC values and must not be estimated or borrowed from the
smaller local `gnn-small` study.

| Metric | `gnn-full` | Full-data XGBoost | Current interpretation |
| --- | ---: | ---: | --- |
| AUROC | **0.888018** | 0.880653 | GNN is +0.007365 numerically; paired CI missing |
| Average precision/AUPRC | Not saved | Not saved | Requires validation probabilities; compare with 0.5423 prevalence baseline |
| Accuracy | Not available | Not available | Requires a threshold |
| Precision/PPV | Not available | Not available | Requires a threshold; prevalence-sensitive |
| Recall/sensitivity | Not available | Not available | Requires a threshold |
| Specificity | Not available | Not available | Requires a threshold |
| NPV | Not available | Not available | Requires a threshold; prevalence-sensitive |
| F1 | Not available | Not available | Requires a threshold |
| Balanced accuracy | Not available | Not available | Requires a threshold |
| MCC | Not available | Not available | Requires a threshold |
| Brier score/log loss | Not saved | Not saved | Requires probabilities; evaluates probability quality |
| Calibration intercept/slope | Not saved | Not saved | Requires probabilities and outcomes |

The validation-only inference run should produce three side-by-side operating-point tables:

1. **Common threshold 0.50:** useful as a raw-probability diagnostic, but only fair if both models
   are similarly calibrated.
2. **Per-model maximum-F1 threshold:** shows each model's best exploratory validation trade-off.
   It is model-selection evidence and must not be reported as untouched test performance.
3. **Matched 95% recall:** compares precision, specificity, NPV, alert rate, and alert reduction at
   the same safety-oriented sensitivity. The 95% target is a research scenario pending clinical
   approval.

After choosing and freezing one threshold policy on 2024 Q1, apply it unchanged to 2024 Q2 once.
The final side-by-side table should report counts (`TP`, `FP`, `TN`, `FN`) as well as rates so the
operational difference is visible.

The validation AUROC improved from 0.8715 at epoch 1 to 0.8880 at epoch 43, a gain of 0.0165. Training stopped at epoch 58 after no new best score for 15 epochs. Mean AUROC across the final 15 epochs was 0.8836 with a standard deviation of 0.0019; the final epoch scored 0.8856. Training loss fell from 0.3683 to 0.3321, a 9.84% reduction. This is consistent with successful optimization and a modest late-training plateau, not obvious catastrophic overfitting.

### 3.2 Class balance and AUPRC interpretation

Serious outcomes are clinically uncommon in the general treated population, but this model is trained within a selected spontaneous-report cohort. In the 2024 Q1 validation reports, 54.23% have a documented serious outcome. Therefore:

- the validation AUPRC of a random ranker would be approximately **0.5423**, not a low rare-event prevalence;
- AUPRC remains required, but it must be reported alongside that prevalence baseline;
- performance in FAERS cannot be translated into population incidence, absolute clinical risk, PPV, or NPV in a hospital without external data and recalibration.

### 3.3 Target semantics

The operational target is:

> `is_serious = 1` when FAERS records any official serious outcome code; otherwise `0`.

The implementation includes death, life-threatening outcome, hospitalization, disability, required intervention, congenital anomaly, and other serious outcome codes. Reports with no OUTC row are retained as zero so that the cohort has a comparison class.

Consequently, the negative class means **“no recorded serious outcome”**, not “verified safe,” “no adverse event,” or “drug did not cause harm.” This distinction must appear in every model card, paper, dashboard, and user interface.

### 3.4 Context from related studies

Direct leaderboard comparisons would be misleading because study populations, labels, predictors, and validation designs differ. The studies below provide context only:

- Martin et al. reported AUROC 0.85 for seriousness assessment in French patient-submitted reports, with consistent external-center results [3].
- Zhao et al. reported LightGBM AUROC 0.806 and AUPRC 0.615 for severe oncology FAERS reports [4].
- Lam et al. reported AUROC 0.76, AUPRC 0.47, PPV 0.50, recall 0.62, and specificity 0.90 for medication harm in a small prospectively collected hospital cohort [7].

TekaRx's 0.8880 AUROC is encouraging relative to these figures, but it does not prove superior clinical performance. TekaRx currently predicts a FAERS documentation label, whereas the compared studies use different populations and reference standards.

## 4. Recommended evaluation framework

The framework below is grounded in the cited prediction-model, CDSS, pharmacovigilance, clinical-utility, and explainability literature. TRIPOD+AI explicitly separates discrimination, calibration, and clinical utility and requires evaluation data to be distinct from model development and selection data [10].

### 4.1 Machine-learning evaluation

| Metric or analysis | Why it matters | Primary evidence | TekaRx status |
| --- | --- | --- | --- |
| AUROC with 95% CI | Threshold-free ranking discrimination | TRIPOD+AI [10]; Martin et al. [3]; Zhao et al. [4] | Point estimate available; CI missing |
| AUPRC with prevalence baseline | Summarizes precision–recall trade-off and is informative when class balance changes | Zhao et al. [4]; Lam et al. [7] | Missing |
| Log loss and Brier score | Measures quality of probabilistic predictions, not just rank order | TRIPOD+AI performance domains [10] | Missing |
| Calibration curve, intercept, and slope | Determines whether predicted probabilities correspond to observed frequencies | TRIPOD+AI [10] | Missing |
| Precision, recall, specificity, NPV, F1 | Defines behavior at an operational threshold | Martin et al. [3]; Skalafouris et al. [2]; Lam et al. [7] | Missing |
| Confusion matrix and counts per 10,000 reports | Makes false-alert and missed-event burden concrete | Skalafouris et al. [2]; Lam et al. [7] | Missing |
| Paired model comparison | Quantifies uncertainty in the GNN–XGBoost difference on identical reports | Required to support a superiority claim | Missing |
| Subgroup metrics and calibration | Detects unequal performance and supports fairness assessment | TRIPOD+AI [10] | Missing |
| Rolling-quarter performance | Measures temporal drift and operational stability | External/temporal evaluation principles [1, 10] | One quarter only |

Recommended uncertainty procedure:

1. Resample patients/reports, not individual drug edges.
2. Use at least 1,000 paired bootstrap samples.
3. Report percentile 95% confidence intervals for AUROC, AUPRC, Brier score, and threshold metrics.
4. Compute the GNN–XGBoost difference inside each paired bootstrap sample.
5. Preserve the observed class prevalence in the untouched evaluation set; do not oversample evaluation data.

### 4.2 Clinical and workflow evaluation

| Metric or analysis | Why it matters | Evidence | TekaRx recommendation |
| --- | --- | --- | --- |
| Alert reduction at fixed recall | Directly measures workload reduction while constraining missed serious reports | CDSS alert-fatigue findings [1, 2]; prospective triage efficiency [5] | Primary workflow endpoint |
| PPV | Fraction of reviewed alerts with the target outcome | CDSS PPV varies widely and is prevalence-sensitive [1, 2] | Report at every proposed threshold |
| NPV | Reliability of the low-priority group | Skalafouris et al. [2] | Report with false negatives per 10,000 |
| Sensitivity and specificity | Separates detection from false-positive control | CDSS and pharmacovigilance studies [1–4, 7] | Mandatory |
| Decision-curve analysis | Tests net benefit against review-all and review-none strategies | Vickers and Elkin [11]; TRIPOD+AI [10]; Lam et al. [7] | Use only over clinically plausible thresholds |
| Time saved and cases reviewed per true positive | Connects model output to operational value | MLIT prospective study [5] | Measure in silent prospective pilot |
| Temporal and external evaluation | Tests drift and transportability | Damoiseaux-Volman et al. [1]; Martin et al. [3]; TRIPOD+AI [10] | Required before deployment |

For a threshold \(t\):

```text
alert_reduction = 1 - number_flagged_at_t / number_of_reports
net_benefit(t) = TP/N - (FP/N) * t/(1-t)
```

The recommended primary operating-point analysis is **maximum alert reduction subject to a clinician-approved minimum recall**. A 95% recall target is a reasonable research scenario to plot, but it is not a literature-mandated safety threshold. Pharmacovigilance and clinical stakeholders must set the acceptable miss rate before the final test is opened.

### 4.3 Explainability evaluation

| Evaluation | Purpose | Evidence | TekaRx implementation target |
| --- | --- | --- | --- |
| Global tabular importance | Identifies influential patient and regimen features | SHAP [12]; Zhao et al. [4] | SHAP for XGBoost baseline; permutation/ablation for GNN inputs |
| Local patient-feature attribution | Explains why one report received a score | SHAP principles [12] | Integrated gradients or feature occlusion on the selected GNN |
| Edge/drug importance | Identifies drug exposures that change a patient score | GNNExplainer [13] | Edge occlusion and GNNExplainer-style subgraph attribution |
| Fidelity | Tests whether keeping/removing highlighted inputs changes the prediction as claimed | Explainability must be evaluated, not merely displayed [6, 13] | Deletion/insertion curves and score change |
| Stability | Tests whether similar cases and repeated runs produce consistent explanations | Trust and reproducibility concerns [5, 6] | Rank correlation across seeds and small perturbations |
| Sparsity | Keeps explanations reviewable | GNNExplainer [13] | Number of features/edges needed for fixed fidelity |
| Clinical plausibility review | Detects spurious but statistically predictive explanations | Expert alignment and human oversight [5, 6] | Blinded pharmacist/pharmacovigilance review |

An explanation is not evidence of causality. FAERS associations and model attributions should be treated as prioritization signals requiring clinical and epidemiological confirmation [6, 8, 9].

### 4.4 Subgroup and robustness evaluation

At minimum, report AUROC, AUPRC, calibration, recall, PPV, sample size, and outcome prevalence for:

- sex and unknown sex;
- age groups and missing age;
- polypharmacy strata;
- reporter/occupation type;
- documented versus missing weight and dosage;
- mapped DrugCentral exposures, top-500 unknown drugs, and the `OTHER` node;
- boxed-warning and high-risk ATC exposure groups;
- malignancy, cardiovascular, and infection indication flags;
- report quarter and, when feasible, reporter country.

Use confidence intervals and avoid ranking very small subgroups. Any threshold proposed for shared deployment should be checked for subgroup sensitivity gaps and calibration drift.

## 5. Leakage-safe evaluation sequence

### Stage A — complete validation diagnostics without touching test

1. Restore the selected epoch-43 checkpoint and exact `gnn-full` graph.
2. Generate and save 2024 Q1 probabilities once.
3. Produce AUROC, AUPRC, Brier score, log loss, calibration plot/intercept/slope, and bootstrap intervals.
4. Compare GNN and XGBoost with paired bootstrap differences.
5. Evaluate candidate alert thresholds, including alert reduction at fixed recall.
6. Run subgroup, missingness, and explanation robustness analyses.
7. Select any probability calibrator and operational threshold using validation only.
8. Freeze the model hash, calibrator, threshold, metric code, and analysis plan.

### Stage B — one-time internal temporal test

Only after Stage A is signed off:

1. Evaluate the frozen package once on 2024 Q2.
2. Do not retune the model, calibrator, feature set, or threshold from test results.
3. Report all prespecified metrics and confidence intervals, including unfavorable results.
4. Mark the test as consumed in the durable model/evaluation manifest.

If the model is changed afterward, the 2024 Q2 set becomes historical evidence, not an untouched test for the revised model.

### Stage C — later-quarter and external evaluation

- Evaluate rolling FAERS quarters after 2024 Q2 to characterize drift.
- Use a genuinely independent pharmacovigilance or clinically adjudicated dataset for external evaluation.
- Recalibrate probabilities for the intended deployment population; FAERS-derived probabilities should not be used as absolute patient risk.

### Stage D — prospective silent-mode workflow evaluation

Before decision support is shown to users:

- run predictions without influencing review decisions;
- measure reviewer agreement, alert reduction, time saved, overrides, and missed clinically meaningful cases;
- conduct blinded clinical plausibility review of explanations;
- define escalation and human-override procedures;
- monitor drift, calibration, and subgroup performance.

## 6. Required evaluation artifacts

The next evaluation implementation should produce immutable, versioned outputs such as:

```text
data/processed/evaluation/
├── gnn_full_validation_predictions.parquet
├── gnn_full_validation_metrics.json
├── gnn_full_threshold_table.parquet
├── gnn_full_subgroup_metrics.parquet
├── gnn_full_bootstrap_intervals.parquet
├── gnn_full_calibration.png
├── gnn_full_roc_pr_curves.png
├── gnn_full_decision_curve.png
├── gnn_full_explanation_audit.parquet
└── gnn_full_evaluation_manifest.json
```

The prediction table should minimally contain `primaryid`, `caseid`, `quarter`, `split`, `y_true`, `y_score`, model/checkpoint IDs, and prespecified subgroup columns. The manifest should record input hashes, code revision, package versions, seed, thresholds, test-consumption state, and every output hash.

## 7. Decision gates

### Gate 1: acceptable for final internal test

- validation probabilities and all core metrics are reproducibly saved;
- discrimination and calibration have bootstrap confidence intervals;
- the threshold and minimum acceptable recall are approved before test access;
- GNN versus XGBoost differences are paired and uncertainty-quantified;
- no unresolved leakage or case-version overlap is found;
- subgroup and explanation audits reveal no unexamined critical failure;
- analysis code and model artifacts are frozen by hash.

### Gate 2: acceptable for prospective research pilot

- the one-time temporal test meets the prespecified operating criteria;
- calibration is acceptable or a frozen recalibration method is applied;
- error review characterizes false negatives and major false-positive clusters;
- later-quarter or independent-data performance is available;
- the tool is explicitly positioned as prioritization support with human oversight.

### Gate 3: acceptable for clinical deployment

Not established by this retrospective FAERS study. It would require prospective workflow evidence, clinically adjudicated outcomes, governance, monitoring, human-factors testing, and applicable regulatory review.

## 8. Important limitations

1. **FAERS cannot establish causation or incidence.** Reports may be incomplete, duplicated, selectively reported, and influenced by publicity or regulation. FDA states that an event report does not establish that a drug caused the event and that report counts cannot estimate occurrence rates [8, 9].
2. **The target is documentation-based.** The model predicts whether a serious outcome code was recorded, not whether a medication caused a serious event.
3. **The negative label is weak.** No recorded serious outcome is not equivalent to verified non-serious or safe.
4. **Validation informed early stopping.** The 2024 Q1 result is a selection metric, not the final unbiased internal test.
5. **Only AUROC is currently available for the GNN.** Probability quality and operational utility remain unknown.
6. **No confidence interval is available.** The precision of 0.8880 and the significance of the +0.0074 baseline difference are unknown.
7. **One seed was evaluated.** Training variability should be assessed across prespecified seeds or deterministic reruns when computationally feasible.
8. **No external validation exists.** A later FAERS quarter is temporal validation, not external validation in another health system or reporting network.
9. **Explainability is not causality.** Feature and edge importance can expose model behavior but cannot confirm biological mechanisms.

## 9. Citation corrections to the supplied literature list

The proposed framework is retained, but several supplied bibliography entries required correction:

| Supplied form | Verified form |
| --- | --- |
| Rommers et al. (2021), DOI `10.1111/bcp.15160` | Damoiseaux-Volman et al. (2022); the DOI and scoping review belong to this paper [1]. Rommers et al. are authors of a different 2013 rule-effectiveness study cited within the literature. |
| Carli et al. (2022), DOI `10.1007/s11096-022-01505-5` | Skalafouris et al. (2023), published online December 2022 [2]. “Carli et al.” is discussed in its literature review, not the author list. |
| Zou et al. (2026), article `e2500081` | Zhao, Wang, Zou, and Ranasinghe (2026), DOI `10.1200/CCI-25-00081` [4]. |
| Medication-harm paper listed as *BMC Medical Informatics and Decision Making* | Lam et al. (2026), *Therapeutic Advances in Drug Safety*, DOI `10.1177/20420986251409325` [7]. |
| “Beyond black boxes” listed as *European Journal of Clinical Pharmacology* | Ferreira-da-Silva et al., *International Journal of Clinical Pharmacy*, DOI `10.1007/s11096-025-02004-z` [6]. |
| FDA (2005/2020) guidance | Final guidance was issued in March 2005; later web-page updates do not change the guidance publication year [9]. |
| WHO manual without year | WHO pharmacovigilance indicators manual, version 1.0, 2015 [14]. |

## 10. References

1. Damoiseaux-Volman BA, Medlock S, van der Meulen DM, et al. Clinical validation of clinical decision support systems for medication review: a scoping review. *British Journal of Clinical Pharmacology*. 2022;88(5):2035–2051. [doi:10.1111/bcp.15160](https://doi.org/10.1111/bcp.15160)
2. Skalafouris C, Blanc A-L, Grosgurin O, et al. Development and retrospective evaluation of a clinical decision support system for the efficient detection of drug-related problems by clinical pharmacists. *International Journal of Clinical Pharmacy*. 2023;45:406–413. [doi:10.1007/s11096-022-01505-5](https://doi.org/10.1007/s11096-022-01505-5)
3. Martin GL, Jouganous J, Savidan R, et al. Validation of artificial intelligence to support the automatic coding of patient adverse drug reaction reports, using nationwide pharmacovigilance data. *Drug Safety*. 2022;45(5):535–548. [doi:10.1007/s40264-022-01153-8](https://doi.org/10.1007/s40264-022-01153-8)
4. Zhao LX, Wang C, Zou J, Ranasinghe P. Machine learning model for predicting severe adverse events in oncology patients using the US Food and Drug Administration Adverse Event Reporting System. *JCO Clinical Cancer Informatics*. 2026;10:e2500081. [doi:10.1200/CCI-25-00081](https://doi.org/10.1200/CCI-25-00081)
5. Ciccarelli L, Mahaux O, Roshan C, et al. Optimising pharmacovigilance efficiency with MLIT (Machine Learning for Intelligent Triage): a tool for statistical safety alerts. *Drug Safety*. Published online 10 August 2026. [doi:10.1007/s40264-026-01696-0](https://doi.org/10.1007/s40264-026-01696-0)
6. Ferreira-da-Silva R, Cruz-Correia R, Ribeiro I. Beyond black boxes: using explainable causal artificial intelligence to separate signal from noise in pharmacovigilance. *International Journal of Clinical Pharmacy*. 2026;48:677–681. [doi:10.1007/s11096-025-02004-z](https://doi.org/10.1007/s11096-025-02004-z)
7. Lam JYJ, Barras M, Scott IA, et al. Machine learning risk prediction models for medication harm in hospitalised adult patients. *Therapeutic Advances in Drug Safety*. 2026;17:20420986251409325. [doi:10.1177/20420986251409325](https://doi.org/10.1177/20420986251409325)
8. US Food and Drug Administration. Understanding CDER's postmarket safety surveillance programs and public data. 2026. [FDA page](https://www.fda.gov/drugs/cder-conversations/understanding-cders-postmarket-safety-surveillance-programs-and-public-data)
9. US Food and Drug Administration. Good pharmacovigilance practices and pharmacoepidemiologic assessment: guidance for industry. March 2005. [FDA guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/good-pharmacovigilance-practices-and-pharmacoepidemiologic-assessment)
10. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)
11. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making*. 2006;26(6):565–574. [doi:10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361)
12. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*. 2017. [Paper](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
13. Ying R, Bourgeois D, You J, Zitnik M, Leskovec J. GNNExplainer: generating explanations for graph neural networks. *Advances in Neural Information Processing Systems 32*. 2019. [Paper](https://arxiv.org/abs/1903.03894)
14. World Health Organization. *WHO pharmacovigilance indicators: a practical manual for the assessment of pharmacovigilance systems*. Version 1.0. 2015. ISBN 9789241508254. [WHO publication](https://www.who.int/publications/i/item/9789241508254)

## 11. Bottom line

`gnn-full` has crossed the bar for **promising temporal discrimination**, not the bar for complete evaluation or deployment. The immediate priority is to save validation probabilities and complete the prespecified discrimination, calibration, threshold, clinical-utility, subgroup, uncertainty, and explainability analyses. Only after those choices are frozen should the 2024 Q2 test be evaluated once.
