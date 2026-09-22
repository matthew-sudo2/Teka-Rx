"""Unit tests for streamlit_app functionality."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add frontend dir to path
sys.path.insert(0, str(Path(__file__).parent))

from streamlit_app import _generate_record, _parse_medications


class TestGenerateRecord(unittest.TestCase):
    """Test the _generate_record function."""

    def test_generate_record_returns_correct_types(self):
        """Verify _generate_record returns correct tuple types."""
        # Create a minimal bundle and dictionary
        bundle = {
            "feature_cols": ["age", "sex_unknown"],
            "imputer": type("obj", (object,), {"statistics_": [50.0, 0.0]})(),
        }
        dictionary = pd.DataFrame(
            {
                "faers_raw": ["ASPIRIN", "IBUPROFEN", "ACETAMINOPHEN"],
                "dc_id": [1, 2, 3],
                "atc_code": ["N02BA01", "M01AE01", "N02BE01"],
                "ror": [1.5, 2.0, 1.2],
                "has_boxed_warning": [0, 0, 0],
            }
        )

        age, sex, meds_str, generated = _generate_record(bundle, dictionary)

        # Verify types
        self.assertIsInstance(age, float)
        self.assertIsInstance(sex, str)
        self.assertIsInstance(meds_str, str)
        self.assertIsInstance(generated, list)

    def test_generate_record_age_in_valid_range(self):
        """Verify generated age is in plausible range."""
        bundle = {
            "feature_cols": ["age"],
            "imputer": type("obj", (object,), {"statistics_": [50.0]})(),
        }
        dictionary = pd.DataFrame(
            {
                "faers_raw": ["ASPIRIN", "IBUPROFEN"],
                "dc_id": [1, 2],
                "atc_code": ["N02BA01", "M01AE01"],
                "ror": [1.5, 2.0],
                "has_boxed_warning": [0, 0],
            }
        )

        for _ in range(10):  # Test multiple times for randomness
            age, _, _, _ = _generate_record(bundle, dictionary)
            self.assertGreaterEqual(age, 30)
            self.assertLessEqual(age, 80)

    def test_generate_record_sex_valid_value(self):
        """Verify generated sex is one of valid options."""
        bundle = {
            "feature_cols": ["sex"],
            "imputer": type("obj", (object,), {"statistics_": [0.0]})(),
        }
        dictionary = pd.DataFrame(
            {
                "faers_raw": ["ASPIRIN"],
                "dc_id": [1],
                "atc_code": ["N02BA01"],
                "ror": [1.5],
                "has_boxed_warning": [0],
            }
        )

        valid_sexes = {"Female", "Male", "Unknown"}
        for _ in range(10):
            _, sex, _, _ = _generate_record(bundle, dictionary)
            self.assertIn(sex, valid_sexes)

    def test_generate_record_meds_not_empty(self):
        """Verify generated medications string is not empty."""
        bundle = {
            "feature_cols": ["age"],
            "imputer": type("obj", (object,), {"statistics_": [50.0]})(),
        }
        dictionary = pd.DataFrame(
            {
                "faers_raw": ["ASPIRIN", "IBUPROFEN", "ACETAMINOPHEN"],
                "dc_id": [1, 2, 3],
                "atc_code": ["N02BA01", "M01AE01", "N02BE01"],
                "ror": [1.5, 2.0, 1.2],
                "has_boxed_warning": [0, 0, 0],
            }
        )

        _, _, meds_str, _ = _generate_record(bundle, dictionary)
        self.assertGreater(len(meds_str), 0)
        # Should have at least one newline (multiple meds) or just med names
        meds_list = meds_str.split("\n")
        self.assertGreaterEqual(len(meds_list), 1)

    def test_generate_record_marked_as_generated(self):
        """Verify generated fields are marked in the returned list."""
        bundle = {
            "feature_cols": ["age"],
            "imputer": type("obj", (object,), {"statistics_": [50.0]})(),
        }
        dictionary = pd.DataFrame(
            {
                "faers_raw": ["ASPIRIN"],
                "dc_id": [1],
                "atc_code": ["N02BA01"],
                "ror": [1.5],
                "has_boxed_warning": [0],
            }
        )

        _, _, _, generated = _generate_record(bundle, dictionary)
        expected_fields = {"age", "sex", "medications"}
        self.assertEqual(set(generated), expected_fields)


class TestParseMedications(unittest.TestCase):
    """Test the _parse_medications function."""

    def test_parse_medications_newline_separated(self):
        """Verify parsing of newline-separated medications."""
        result = _parse_medications("Aspirin\nIbuprofen\nAcetaminophen")
        self.assertEqual(result, ["Aspirin", "Ibuprofen", "Acetaminophen"])

    def test_parse_medications_comma_separated(self):
        """Verify parsing of comma-separated medications."""
        result = _parse_medications("Aspirin, Ibuprofen, Acetaminophen")
        self.assertEqual(result, ["Aspirin", "Ibuprofen", "Acetaminophen"])

    def test_parse_medications_mixed_separators(self):
        """Verify parsing of mixed separator medications."""
        result = _parse_medications("Aspirin, Ibuprofen\nAcetaminophen")
        self.assertEqual(result, ["Aspirin", "Ibuprofen", "Acetaminophen"])

    def test_parse_medications_strips_whitespace(self):
        """Verify whitespace is stripped from parsed medications."""
        result = _parse_medications("  Aspirin  ,  Ibuprofen  ,  Acetaminophen  ")
        self.assertEqual(result, ["Aspirin", "Ibuprofen", "Acetaminophen"])

    def test_parse_medications_empty_string(self):
        """Verify parsing empty string returns empty list."""
        result = _parse_medications("")
        self.assertEqual(result, [])

    def test_parse_medications_only_whitespace(self):
        """Verify parsing whitespace-only string returns empty list."""
        result = _parse_medications("   \n\n   ,  ,  ")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
