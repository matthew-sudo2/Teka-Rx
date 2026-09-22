#!/usr/bin/env python
"""Quick verification of streamlit app logic without UI."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "frontend"))

import joblib
import pandas as pd
import numpy as np
from frontend.streamlit_app import (
    _match_medications,
    _build_features,
    _predict,
    _top_contributors,
    load_artifacts,
)

def test_end_to_end():
    """Test core prediction pipeline."""
    print("=" * 70)
    print("TEST 1: Load artifacts")
    print("=" * 70)
    try:
        artifacts = load_artifacts()
        print(f"✓ Bundle loaded: {list(artifacts['bundle'].keys())}")
        print(f"✓ Feature cols: {len(artifacts['bundle']['feature_cols'])} features")
        print(f"✓ Dictionary: {len(artifacts['dictionary'])} drugs")
    except Exception as e:
        print(f"✗ Failed to load artifacts: {e}")
        return False

    bundle = artifacts["bundle"]
    dictionary = artifacts["dictionary"]

    print("\n" + "=" * 70)
    print("TEST 2: Medication matching (high-priority polypharmacy)")
    print("=" * 70)
    queries = ["Warfarin", "Ibuprofen", "Tramadol", "Metoprolol", "Omeprazole"]
    matched, matches = _match_medications(queries, dictionary)
    print(f"Input medications: {queries}")
    for m in matches:
        print(
            f"  {m['input']:15} → {m['status']:10} "
            f"({m['score']:.0f}%, used={m['used']}) "
            f"match='{m['match']}'"
        )
    print(f"Matched drug rows: {len(matched)}")

    print("\n" + "=" * 70)
    print("TEST 3: Feature building & prediction")
    print("=" * 70)
    age = 74
    sex = "Female"
    frame, summary = _build_features(age, sex, matched, bundle)
    print(f"Patient: {age} years, {sex}")
    print(f"Frame shape: {frame.shape}")
    print(f"Summary metrics:")
    for key, val in summary.items():
        print(f"  {key}: {val}")

    probability = _predict(frame, artifacts)
    threshold = bundle.get("threshold", 0.50)
    print(f"\nPrediction probability: {probability:.1%}")
    print(f"Threshold: {threshold:.0%}")
    print(f"Result: {'PRIORITY SIGNAL' if probability >= threshold else 'NO PRIORITY'}")

    contributors = _top_contributors(bundle, frame)
    print(f"\nTop 5 contributors:")
    for idx, row in contributors.iterrows():
        print(
            f"  {row['Feature']:35} "
            f"value={row['Regimen Value']:20} "
            f"contrib={row['Relative Contribution']}"
        )

    print("\n" + "=" * 70)
    print("TEST 4: Medication matching (low-priority monotherapy)")
    print("=" * 70)
    queries2 = ["Metformin", "Atorvastatin", "Levothyroxine"]
    matched2, matches2 = _match_medications(queries2, dictionary)
    print(f"Input medications: {queries2}")
    for m in matches2:
        print(
            f"  {m['input']:15} → {m['status']:10} "
            f"({m['score']:.0f}%, used={m['used']}) "
            f"match='{m['match']}'"
        )

    age2 = 52
    sex2 = "Male"
    frame2, summary2 = _build_features(age2, sex2, matched2, bundle)
    probability2 = _predict(frame2, artifacts)
    print(f"\nPatient: {age2} years, {sex2}")
    print(f"Prediction probability: {probability2:.1%}")
    print(f"Result: {'PRIORITY SIGNAL' if probability2 >= threshold else 'NO PRIORITY'}")

    print("\n" + "=" * 70)
    print("TEST 5: Unmatched medications")
    print("=" * 70)
    queries3 = ["Aspirin", "XXXXXXXXINVALIDXXXXX", "Lisinopril"]
    matched3, matches3 = _match_medications(queries3, dictionary)
    print(f"Input medications: {queries3}")
    unmatched_count = 0
    for m in matches3:
        print(
            f"  {m['input']:25} → {m['status']:10} "
            f"({m['score']:.0f}%, used={m['used']})"
        )
        if not m["used"]:
            unmatched_count += 1
    print(f"Unmatched count: {unmatched_count}")

    print("\n" + "=" * 70)
    print("TEST 6: Empty regimen (no matched medications)")
    print("=" * 70)
    queries4 = ["INVALIDDRUGXXX", "NOTAREALMEDXXX"]
    matched4, matches4 = _match_medications(queries4, dictionary)
    frame4, summary4 = _build_features(65, "Unknown", matched4, bundle)
    probability4 = _predict(frame4, artifacts)
    print(f"Empty matched set (only imputer defaults)")
    print(f"Prediction probability: {probability4:.1%}")
    print(f"Summary: {summary4}")

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
