"""Validation-only ablation for normalized FAERS dosage features."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tekarx.transform.graph import (
    NORMALIZED_DOSAGE_FEATURE_PREFIX,
    NORMALIZED_DOSAGE_PATIENT_FEATURES,
    binary_auc,
)


class DosageAblationError(RuntimeError):
    """Raised when a dosage ablation artifact or invariant is invalid."""


@dataclass(frozen=True)
class DosageAblationRecord:
    """Summary of an apples-to-apples normalized-dosage ablation."""

    tabular_path: str
    with_dosage_model_path: str
    without_dosage_model_path: str
    validation_predictions_path: str
    manifest_path: str
    patient_rows: int
    training_rows: int
    validation_rows: int
    feature_count_with_dosage: int
    feature_count_without_dosage: int
    dosage_feature_count: int
    best_iteration_with_dosage: int
    best_iteration_without_dosage: int
    validation_auc_with_dosage: float
    validation_auc_without_dosage: float
    validation_auc_delta: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_probability_positive: float
    bootstrap_samples: int
    test_evaluated: bool


def evaluate_dosage_ablation(
    *,
    data_dir: Path,
    tabular_path: Path | None = None,
    xgb_rounds: int = 1000,
    xgb_early_stopping: int = 50,
    xgb_max_depth: int = 0,
    xgb_max_leaves: int = 63,
    threads: int | None = None,
    seed: int = 42,
    bootstrap_samples: int = 500,
) -> DosageAblationRecord:
    """Retrain matched XGBoost models and compare validation AUC only.

    The control removes the reviewed ``dose_normalized_`` package. It includes
    relative amount/daily-dose statistics, availability and scheduling coverage,
    high-dose flags, and parenteral indicators. Existing coarse exposure fields
    remain in both arms, so the delta measures the package's incremental value.
    Test labels are never read or scored.
    """
    if xgb_rounds < 1 or xgb_early_stopping < 1:
        raise ValueError("XGBoost rounds and early stopping must be positive")
    if threads is not None and threads < 1:
        raise ValueError("threads must be positive")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise DosageAblationError(
            'missing XGBoost; install with python -m pip install -e ".[graph]"'
        ) from exc

    tabular_path = tabular_path or data_dir / "processed" / "tekarx_tabular_baseline.npz"
    if not tabular_path.is_file():
        raise DosageAblationError(f"missing tabular graph artifact: {tabular_path}")

    with np.load(tabular_path, allow_pickle=False) as archive:
        required = {"X", "y", "train_mask", "validation_mask", "feature_names"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise DosageAblationError(f"tabular artifact is missing arrays: {missing}")
        patient_x = np.asarray(archive["X"], dtype=np.float32)
        labels = np.asarray(archive["y"], dtype=np.int8)
        train_mask = np.asarray(archive["train_mask"], dtype=bool)
        validation_mask = np.asarray(archive["validation_mask"], dtype=bool)
        feature_names = tuple(str(name) for name in archive["feature_names"].tolist())
        primaryid = (
            np.asarray(archive["primaryid"])
            if "primaryid" in archive.files
            else np.arange(patient_x.shape[0])
        )

    _validate_ablation_arrays(
        patient_x=patient_x,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        feature_names=feature_names,
    )
    dosage_indices = tuple(
        index
        for index, name in enumerate(feature_names)
        if name in NORMALIZED_DOSAGE_PATIENT_FEATURES
    )
    if not dosage_indices:
        raise DosageAblationError(
            f"no {NORMALIZED_DOSAGE_FEATURE_PREFIX!r} features were found; "
            "rebuild the enriched cohort and graph first"
        )
    dosage_index_set = set(dosage_indices)
    without_indices = tuple(
        index for index in range(len(feature_names)) if index not in dosage_index_set
    )
    if not without_indices:
        raise DosageAblationError("normalized dosage features cannot be the entire matrix")

    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    common_parameters = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": xgb_max_depth,
        "max_leaves": xgb_max_leaves,
        "grow_policy": "lossguide" if xgb_max_leaves else "depthwise",
        "eta": 0.03 if xgb_max_leaves else 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": seed,
        "nthread": threads or 0,
    }
    without_path = output_dir / "xgboost_without_normalized_dosage.json"
    without_auc, without_gain, without_scores, without_best_iteration = _train_one_arm(
        xgb=xgb,
        patient_x=patient_x,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        feature_names=tuple(feature_names[index] for index in without_indices),
        feature_indices=without_indices,
        parameters=common_parameters,
        rounds=xgb_rounds,
        early_stopping=xgb_early_stopping,
        model_path=without_path,
    )
    with_path = output_dir / "xgboost_with_normalized_dosage.json"
    with_auc, with_gain, with_scores, with_best_iteration = _train_one_arm(
        xgb=xgb,
        patient_x=patient_x,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        feature_names=feature_names,
        feature_indices=tuple(range(len(feature_names))),
        parameters=common_parameters,
        rounds=xgb_rounds,
        early_stopping=xgb_early_stopping,
        model_path=with_path,
    )
    validation_labels = labels[validation_mask]
    delta_samples = paired_stratified_bootstrap_auc_delta(
        validation_labels,
        with_scores,
        without_scores,
        samples=bootstrap_samples,
        seed=seed,
    )
    ci_lower, ci_upper = np.quantile(delta_samples, [0.025, 0.975]).tolist()
    probability_positive = float(np.mean(delta_samples > 0.0))
    predictions_path = output_dir / "dosage_ablation_validation_predictions.npz"
    predictions_temporary = predictions_path.with_suffix(".npz.tmp")
    with predictions_temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            validation_row_index=np.flatnonzero(validation_mask),
            primaryid=primaryid[validation_mask],
            y=validation_labels,
            score_with_normalized_dosage=with_scores,
            score_without_normalized_dosage=without_scores,
            paired_bootstrap_auc_delta=delta_samples,
        )
    os.replace(predictions_temporary, predictions_path)

    manifest_path = output_dir / "dosage_ablation_manifest.json"
    record = DosageAblationRecord(
        tabular_path=str(tabular_path),
        with_dosage_model_path=str(with_path),
        without_dosage_model_path=str(without_path),
        validation_predictions_path=str(predictions_path),
        manifest_path=str(manifest_path),
        patient_rows=int(patient_x.shape[0]),
        training_rows=int(train_mask.sum()),
        validation_rows=int(validation_mask.sum()),
        feature_count_with_dosage=len(feature_names),
        feature_count_without_dosage=len(without_indices),
        dosage_feature_count=len(dosage_indices),
        best_iteration_with_dosage=with_best_iteration,
        best_iteration_without_dosage=without_best_iteration,
        validation_auc_with_dosage=with_auc,
        validation_auc_without_dosage=without_auc,
        validation_auc_delta=with_auc - without_auc,
        bootstrap_ci_lower=float(ci_lower),
        bootstrap_ci_upper=float(ci_upper),
        bootstrap_probability_positive=probability_positive,
        bootstrap_samples=bootstrap_samples,
        test_evaluated=False,
    )
    _write_ablation_manifest(
        manifest_path,
        record=record,
        feature_names=feature_names,
        dosage_feature_names=tuple(feature_names[index] for index in dosage_indices),
        parameters=common_parameters,
        rounds=xgb_rounds,
        early_stopping=xgb_early_stopping,
        with_gain=with_gain,
        without_gain=without_gain,
    )
    print(
        "Dosage ablation validation AUC: "
        f"without={without_auc:.6f}, with={with_auc:.6f}, delta={with_auc - without_auc:+.6f}"
    )
    print(
        f"Paired stratified bootstrap 95% CI: [{ci_lower:+.6f}, {ci_upper:+.6f}] "
        f"(P(delta > 0)={probability_positive:.3f})"
    )
    print("Test labels were not evaluated; the final test split remains untouched.")
    return record


def _train_one_arm(
    *,
    xgb: object,
    patient_x: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    feature_names: tuple[str, ...],
    feature_indices: tuple[int, ...],
    parameters: dict[str, object],
    rounds: int,
    early_stopping: int,
    model_path: Path,
) -> tuple[float, dict[str, float], np.ndarray, int]:
    train_values = np.ascontiguousarray(patient_x[train_mask][:, feature_indices])
    validation_values = np.ascontiguousarray(patient_x[validation_mask][:, feature_indices])
    train_matrix = xgb.DMatrix(
        train_values,
        label=labels[train_mask],
        feature_names=list(feature_names),
    )
    validation_matrix = xgb.DMatrix(
        validation_values,
        label=labels[validation_mask],
        feature_names=list(feature_names),
    )
    del train_values, validation_values
    booster = xgb.train(
        parameters,
        train_matrix,
        num_boost_round=rounds,
        evals=[(validation_matrix, "validation")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", rounds - 1))
    scores = booster.predict(validation_matrix, iteration_range=(0, best_iteration + 1))
    auc = binary_auc(labels[validation_mask], scores)
    gain = {
        name: float(value)
        for name, value in sorted(
            booster.get_score(importance_type="gain").items(),
            key=lambda item: (-item[1], item[0]),
        )
    }
    temporary = model_path.with_name(f"{model_path.stem}.tmp{model_path.suffix}")
    booster.save_model(temporary)
    os.replace(temporary, model_path)
    result_scores = np.asarray(scores, dtype=np.float32)
    del booster, train_matrix, validation_matrix, scores
    gc.collect()
    return auc, gain, result_scores, best_iteration


def paired_stratified_bootstrap_auc_delta(
    labels: np.ndarray,
    scores_with_dosage: np.ndarray,
    scores_without_dosage: np.ndarray,
    *,
    samples: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Return paired, class-stratified bootstrap draws of the AUC difference."""
    y = np.asarray(labels, dtype=np.int8)
    with_scores = np.asarray(scores_with_dosage, dtype=np.float64)
    without_scores = np.asarray(scores_without_dosage, dtype=np.float64)
    if y.ndim != 1 or with_scores.shape != y.shape or without_scores.shape != y.shape:
        raise DosageAblationError("bootstrap labels and scores must be aligned vectors")
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not positive.size or not negative.size:
        raise DosageAblationError("bootstrap requires both target classes")
    generator = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    resampled_labels = np.concatenate(
        (np.ones(positive.size, dtype=np.int8), np.zeros(negative.size, dtype=np.int8))
    )
    for sample in range(samples):
        indices = np.concatenate(
            (
                generator.choice(positive, size=positive.size, replace=True),
                generator.choice(negative, size=negative.size, replace=True),
            )
        )
        deltas[sample] = _vectorized_binary_auc(
            resampled_labels, with_scores[indices]
        ) - _vectorized_binary_auc(resampled_labels, without_scores[indices])
    return deltas


def _vectorized_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware rank AUC with grouping performed in NumPy for bootstrap speed."""
    y = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise DosageAblationError("AUC requires both target classes")
    order = np.argsort(predictions, kind="mergesort")
    sorted_scores = predictions[order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.asarray([y.size], dtype=np.int64)))
    average_ranks = (starts + 1 + ends) / 2.0
    positive_per_tie_group = np.add.reduceat(y[order].astype(np.float64), starts)
    positive_rank_sum = float(np.dot(positive_per_tie_group, average_ranks))
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def _validate_ablation_arrays(
    *,
    patient_x: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    feature_names: tuple[str, ...],
) -> None:
    if patient_x.ndim != 2:
        raise DosageAblationError("X must be a two-dimensional matrix")
    row_count = patient_x.shape[0]
    if any(values.shape != (row_count,) for values in (labels, train_mask, validation_mask)):
        raise DosageAblationError("labels and split masks must match X rows")
    if patient_x.shape[1] != len(feature_names):
        raise DosageAblationError("feature_names must match X columns")
    if not np.isfinite(patient_x).all():
        raise DosageAblationError("X contains NaN or infinity")
    if not train_mask.any() or not validation_mask.any():
        raise DosageAblationError("training and validation masks must be non-empty")
    if np.any(train_mask & validation_mask):
        raise DosageAblationError("training and validation masks overlap")
    for split_name, mask in (("training", train_mask), ("validation", validation_mask)):
        values = labels[mask]
        if not (np.any(values == 0) and np.any(values == 1)):
            raise DosageAblationError(f"{split_name} split must contain both target classes")


def _write_ablation_manifest(
    path: Path,
    *,
    record: DosageAblationRecord,
    feature_names: tuple[str, ...],
    dosage_feature_names: tuple[str, ...],
    parameters: dict[str, object],
    rounds: int,
    early_stopping: int,
    with_gain: dict[str, float],
    without_gain: dict[str, float],
) -> None:
    without_names = [name for name in feature_names if name not in dosage_feature_names]
    payload = {
        "study": "incremental normalized dosage package ablation",
        "comparison": (
            "identical train/validation rows and hyperparameters; control removes the "
            f"reviewed features prefixed {NORMALIZED_DOSAGE_FEATURE_PREFIX!r}, including "
            "relative dose, availability, scheduling, high-dose, and parenteral features; "
            "existing coarse exposure features remain in both arms"
        ),
        "decision_rule": (
            "advance normalized dosage to the full build only if the paired bootstrap "
            "95% CI lower bound is above zero; otherwise treat the result as inconclusive"
        ),
        "test_protocol": {
            "evaluated": False,
            "predictions_generated": False,
            "labels_used_for_training_or_selection": False,
        },
        "feature_sets": {
            "with_normalized_dosage": list(feature_names),
            "without_normalized_dosage": without_names,
            "normalized_dosage_only": list(dosage_feature_names),
        },
        "xgboost": {
            "parameters": parameters,
            "num_boost_round": rounds,
            "early_stopping_rounds": early_stopping,
        },
        "gain": {
            "with_normalized_dosage": with_gain,
            "without_normalized_dosage": without_gain,
        },
        "record": asdict(record),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
