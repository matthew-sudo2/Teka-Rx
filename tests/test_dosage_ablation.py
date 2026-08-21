import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")

from tekarx.modeling.dosage_ablation import (  # noqa: E402
    evaluate_dosage_ablation,
    paired_stratified_bootstrap_auc_delta,
)


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    better = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    worse = np.asarray([0.9, 0.2, 0.7, 0.3, 0.8, 0.1])
    first = paired_stratified_bootstrap_auc_delta(
        labels, better, worse, samples=20, seed=7
    )
    second = paired_stratified_bootstrap_auc_delta(
        labels, better, worse, samples=20, seed=7
    )
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0.0)


def test_xgboost_dosage_ablation_never_outputs_test_predictions(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    train_rows = 40
    validation_rows = 20
    test_rows = 10
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 35)
    dosage_signal = labels.astype(np.float32)
    patient_x = np.column_stack(
        (
            np.zeros(labels.size, dtype=np.float32),
            np.ones(labels.size, dtype=np.float32),
            dosage_signal,
        )
    )
    train_mask = np.zeros(labels.size, dtype=bool)
    train_mask[:train_rows] = True
    validation_mask = np.zeros(labels.size, dtype=bool)
    validation_mask[train_rows : train_rows + validation_rows] = True
    test_mask = ~(train_mask | validation_mask)
    assert int(test_mask.sum()) == test_rows
    tabular = processed / "tekarx_tabular_baseline.npz"
    np.savez_compressed(
        tabular,
        X=patient_x,
        y=labels,
        primaryid=np.asarray([f"p{index}" for index in range(labels.size)]),
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        feature_names=np.asarray(
            [
                "age_over_120",
                "exposure_dose_documented_fraction",
                "dose_normalized_relative_log_max",
            ]
        ),
    )

    record = evaluate_dosage_ablation(
        data_dir=data_dir,
        xgb_rounds=5,
        xgb_early_stopping=2,
        threads=1,
        bootstrap_samples=10,
    )

    assert record.feature_count_with_dosage == 3
    assert record.feature_count_without_dosage == 2
    assert record.test_evaluated is False
    predictions = np.load(record.validation_predictions_path, allow_pickle=False)
    assert predictions["y"].shape == (validation_rows,)
    assert "test" not in " ".join(predictions.files).lower()
    manifest = json.loads(Path(record.manifest_path).read_text(encoding="utf-8"))
    assert manifest["test_protocol"] == {
        "evaluated": False,
        "labels_used_for_training_or_selection": False,
        "predictions_generated": False,
    }
    assert manifest["feature_sets"]["normalized_dosage_only"] == [
        "dose_normalized_relative_log_max"
    ]
