"""Focused regressions for the external full19 scoring bridge."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from intervention_analysis import external_scoring_bridge as bridge

WEIGHT_DIR = bridge.DEFAULT_WEIGHT_DIR


class ExternalScoringBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bridge = bridge
        cls.model = cls.bridge.load_frozen_linear_model(WEIGHT_DIR)

    def test_linear_score_and_zero_imputation(self) -> None:
        frame = pd.DataFrame({"sample_id": ["observed", "missing"]})
        for feature in self.model.features:
            frame[feature] = 0.0
        frame.loc[0, self.model.features[0]] = 2.0
        frame.loc[1, self.model.features[0]] = np.nan
        observed = self.bridge.score_components(frame, self.model).predicted_bioage
        expected = [
            self.model.intercept + 2.0 * self.model.coefficients[0],
            self.model.intercept,
        ]
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-13)

    def test_paired_difference_sd_and_declared_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            shard = run / "shards" / "geo__synthetic"
            shard.mkdir(parents=True)
            task = {
                "task_key": "geo__synthetic",
                "kind": "geo",
                "public_id": "GSEsynthetic",
                "retained_families": ["geo::welch", "geo::paired"],
            }
            (run / "tasks_initial.json").write_text(json.dumps([task]))
            (run / "tasks_recovered.json").write_text("[]\n")
            component_rows, design_rows = [], []
            definitions = {
                "welch": ("donor_level_welch", [1, 2, 3, 4], [4, 5, 6, 7]),
                "paired": ("donor_level_paired", [10, 11, 12, 13], [11, 13, 15, 17]),
            }
            for comparison, (method, control, experiment) in definitions.items():
                for group, values in (("control", control), ("experiment", experiment)):
                    for index, target in enumerate(values):
                        sample = f"{comparison}_{group}_{index}"
                        donor = (
                            f"{comparison}_{index}"
                            if method == "donor_level_paired"
                            else f"{comparison}_{group}_{index}"
                        )
                        components = {feature: 0.0 for feature in self.model.features}
                        components.update(sample_id=sample)
                        components[self.model.features[0]] = (
                            target - self.model.intercept
                        ) / self.model.coefficients[0]
                        component_rows.append(components)
                        design_rows.append(
                            {
                                "comparison_id": comparison,
                                "gse_id": "GSEsynthetic",
                                "comparison_name": comparison,
                                "sample_id": sample,
                                "group": group,
                                "donor_id": donor,
                                "chronological_age": 50.0,
                                "analysis_method": method,
                                "age_correction_stratum": comparison,
                                "canonical_comparison_id": comparison,
                            }
                        )
            pd.DataFrame(component_rows).to_csv(shard / "component_predictions_19.csv", index=False)
            pd.DataFrame(design_rows).to_csv(shard / "analysis_sample_predictions.csv", index=False)
            base = {
                "public_id": "GSEsynthetic",
                "source": "synthetic",
                "source_dataset_id": "GSEsynthetic",
            }
            panel = pd.DataFrame(
                [
                    base
                    | {
                        "family_test_id": "geo::welch",
                        "comparison_id": "welch",
                        "intervention": "welch",
                        "display_intervention": "Welch",
                        "analysis_method": "donor_level_welch",
                        "test": "two_sided_welch_t_test",
                    },
                    base
                    | {
                        "family_test_id": "geo::paired",
                        "comparison_id": "paired",
                        "intervention": "paired",
                        "display_intervention": "Paired",
                        "analysis_method": "donor_level_paired",
                        "test": "two_sided_paired_t_test",
                    },
                ]
            )
            moments = self.bridge.compute_moments(panel, run, WEIGHT_DIR).set_index(
                "family_test_id"
            )
            self.assertAlmostEqual(moments.loc["geo::welch", "control_mean"], -1.5)
            self.assertAlmostEqual(moments.loc["geo::welch", "experiment_mean"], 1.5)
            self.assertTrue(np.isnan(moments.loc["geo::welch", "paired_difference_sd"]))
            self.assertAlmostEqual(moments.loc["geo::paired", "control_mean"], -1.25)
            self.assertAlmostEqual(moments.loc["geo::paired", "experiment_mean"], 1.25)
            self.assertAlmostEqual(
                moments.loc["geo::paired", "paired_difference_sd"],
                np.std([1, 2, 3, 4], ddof=1),
            )


if __name__ == "__main__":
    unittest.main()
