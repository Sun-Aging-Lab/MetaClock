"""Regression tests for dataset shard cache validation and GEO summaries."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from intervention_analysis import run_biometaclock_intervention_analysis as runner
from intervention_analysis.run_geo_biometaclock_gse import summarize_comparisons


class DatasetShardCacheTests(unittest.TestCase):
    """Check cache reuse against small, explicit manifests and prediction tables."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(
            prefix=".test_dataset_cache_", dir=runner.PROJECT_ROOT
        )
        self.root = Path(self.temp.name)
        self.shard = self.root / "shard"
        self.shard.mkdir()
        self.input_paths = [self.root / "sample_manifest.csv", self.root / "matrix.csv"]
        for path in self.input_paths:
            path.write_text("fixture\n", encoding="utf-8")
            os.utime(path, ns=(1_650_000_000_000_000_000,) * 2)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.model_weights = self.model_dir / "weights.csv"
        self.model_metadata = self.model_dir / "metadata.json"
        self.model_weights.write_text("fixture weights\n", encoding="utf-8")
        self.model_metadata.write_text("{}\n", encoding="utf-8")
        self.model_parameters = {
            "features": ["clock_a", "clock_b"],
            "coefficients": [0.25, -0.5],
            "intercept": 2.0,
        }
        self.model_loader = patch.object(
            runner, "load_published_biometaclock", side_effect=self._load_model
        )
        self.model_loader.start()
        self.addCleanup(self.model_loader.stop)
        self.addCleanup(self.temp.cleanup)
        self.expected_samples = pd.DataFrame(
            [
                {"comparison_id": comparison, "gse_id": "GSE123", "sample_id": sample,
                 "donor_id": donor, "group": group}
                for comparison in ("cmp_a", "cmp_b")
                for sample, donor, group in (
                    ("gsm_1", "donor_1", "control"),
                    ("gsm_2", "donor_2", "experiment"),
                )
            ]
        )
        self.expected_ids = {"cmp_a", "cmp_b"}
        self._write_valid_cache()

    def _load_model(self, model_dir: Path):
        model = SimpleNamespace(
            weights_path=self.model_weights,
            metadata_path=self.model_metadata,
            parameters=copy.deepcopy(self.model_parameters),
        )
        return model, list(model.parameters["features"])

    def _write_csv(self, name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(self.shard / name, index=False)

    def _write_valid_cache(self) -> None:
        results = pd.DataFrame(
            [
                {"comparison_id": comparison, "gse_id": "GSE123", "two_sided_p_value": 0.5}
                for comparison in ("cmp_a", "cmp_b")
            ]
        )
        predictions = pd.DataFrame(
            [
                {"gse_id": "GSE123", "sample_id": sample, "predicted_bioage": value}
                for sample, value in (("gsm_1", 60.0), ("gsm_2", 61.0))
            ]
        )
        samples = self.expected_samples.copy()
        summary = {
            "cache_format_version": 1,
            "input_files": sorted(runner.relative_path(path) for path in self.input_paths),
            "started_at_ns": 1_700_000_000_000_000_000,
            "model_parameters": copy.deepcopy(self.model_parameters),
            "model_weights": runner.relative_path(self.model_weights),
            "model_metadata": runner.relative_path(self.model_metadata),
            "beta_distribution_qc_applied": True,
            "gse_id": "GSE123",
        }
        (self.shard / "run_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        self._write_csv("comparison_results.csv", results)
        self._write_csv("sample_predictions.csv", predictions)
        self._write_csv("analysis_sample_predictions.csv", samples)

    def _is_valid(self, expected_samples: pd.DataFrame | None = None) -> bool:
        return runner.completed_shard_is_valid(
            self.shard,
            self.expected_ids,
            self.model_weights,
            dataset_id="GSE123",
            input_paths=self.input_paths,
            expected_samples=(
                self.expected_samples if expected_samples is None else expected_samples
            ),
        )

    def _read_csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.shard / name, dtype=str)

    def test_valid_cache_reuses_csvs_independent_of_row_order(self) -> None:
        for name in (
            "comparison_results.csv",
            "sample_predictions.csv",
            "analysis_sample_predictions.csv",
        ):
            frame = self._read_csv(name)
            self._write_csv(name, frame.iloc[::-1].reset_index(drop=True))
        expected = self.expected_samples.iloc[::-1].reset_index(drop=True)
        self.assertTrue(self._is_valid(expected))

    def test_missing_or_extra_comparison_id_invalidates_cache(self) -> None:
        results = self._read_csv("comparison_results.csv")
        for changed in (results.iloc[:1], pd.concat([
            results,
            pd.DataFrame([{"comparison_id": "cmp_extra", "gse_id": "GSE123",
                           "two_sided_p_value": "0.1"}]),
        ], ignore_index=True)):
            with self.subTest(comparison_ids=changed.comparison_id.tolist()):
                self._write_csv("comparison_results.csv", changed)
                self.assertFalse(self._is_valid())
                self._write_valid_cache()

    def test_missing_or_extra_sample_invalidates_cache(self) -> None:
        predictions = self._read_csv("sample_predictions.csv")
        samples = self._read_csv("analysis_sample_predictions.csv")
        cases = [
            (predictions.iloc[:1], samples.loc[samples.sample_id.ne("gsm_2")]),
            (
                pd.concat([predictions, pd.DataFrame([{
                    "gse_id": "GSE123", "sample_id": "gsm_extra", "predicted_bioage": "62"
                }])], ignore_index=True),
                pd.concat([samples, pd.DataFrame([{
                    "comparison_id": "cmp_a", "gse_id": "GSE123", "sample_id": "gsm_extra",
                    "donor_id": "donor_3", "group": "control",
                }])], ignore_index=True),
            ),
        ]
        for prediction_rows, analysis_rows in cases:
            with self.subTest(n_predictions=len(prediction_rows)):
                self._write_csv("sample_predictions.csv", prediction_rows)
                self._write_csv("analysis_sample_predictions.csv", analysis_rows)
                self.assertFalse(self._is_valid())
                self._write_valid_cache()

    def test_group_change_in_expected_manifest_invalidates_cache(self) -> None:
        expected = self.expected_samples.copy()
        expected.loc[0, "group"] = "experiment"
        self.assertFalse(self._is_valid(expected))

    def test_duplicate_prediction_sample_is_rejected(self) -> None:
        predictions = self._read_csv("sample_predictions.csv")
        self._write_csv(
            "sample_predictions.csv",
            pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True),
        )
        self.assertFalse(self._is_valid())

    def test_changed_model_coefficients_or_intercept_invalidates_cache(self) -> None:
        self.model_parameters["coefficients"][0] += 0.1
        self.assertFalse(self._is_valid())
        self.model_parameters["coefficients"][0] = 0.25
        self.model_parameters["intercept"] += 0.1
        self.assertFalse(self._is_valid())

    def test_input_modified_after_shard_started_invalidates_cache(self) -> None:
        summary_path = self.shard / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        os.utime(self.input_paths[0], ns=(summary["started_at_ns"] + 1,
                                          summary["started_at_ns"] + 1))
        self.assertFalse(self._is_valid())

    def test_missing_cache_format_version_invalidates_legacy_cache(self) -> None:
        summary_path = self.shard / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        del summary["cache_format_version"]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        self.assertFalse(self._is_valid())

    def test_missing_prediction_csv_invalidates_cache(self) -> None:
        (self.shard / "sample_predictions.csv").unlink()
        self.assertFalse(self._is_valid())


class GeoSummarySignatureRegressionTests(unittest.TestCase):
    def test_signature_column_does_not_change_donor_level_statistics(self) -> None:
        rows = []
        differences = np.array([1.0, 2.0, 3.0, 4.0])
        controls = np.array([10.0, 11.0, 13.0, 12.0])
        for donor_index, (control, difference) in enumerate(zip(controls, differences), 1):
            for group, value in (("control", control), ("experiment", control + difference)):
                rows.append({
                    "comparison_id": "cmp_paired",
                    "gse_id": "GSE123",
                    "comparison_name": "synthetic paired comparison",
                    "canonical_comparison_id": "cmp_paired",
                    "analysis_method": "donor_level_paired_no_age_correction",
                    "group": group,
                    "donor_id": f"donor_{donor_index}",
                    "sample_id": f"gsm_{donor_index}_{group}",
                    "chronological_age": float(60 + donor_index),
                    "predicted_bioage": float(value),
                    "analysis_predicted_age": float(value),
                    # Different values ensure this legacy bookkeeping field is
                    # not accidentally retained as a statistical grouping key.
                    "sample_signature_id": f"signature_{donor_index}_{group}",
                })
        with_signature = pd.DataFrame(rows)
        without_signature = with_signature.drop(columns="sample_signature_id")

        results_with, donor_with = summarize_comparisons(with_signature)
        results_without, donor_without = summarize_comparisons(without_signature)

        pd.testing.assert_frame_equal(donor_with, donor_without)
        pd.testing.assert_frame_equal(results_with, results_without)
        result = results_with.iloc[0]
        expected_test = stats.ttest_rel(controls + differences, controls)
        self.assertEqual(result["gse_id"], "GSE123")
        self.assertAlmostEqual(result["mean_diff_experiment_minus_control"], 2.5)
        self.assertAlmostEqual(result["test_statistic_experiment_minus_control"],
                               float(expected_test.statistic))
        self.assertAlmostEqual(result["two_sided_p_value"], float(expected_test.pvalue))


if __name__ == "__main__":
    unittest.main()
