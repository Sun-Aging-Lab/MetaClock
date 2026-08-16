"""Small deterministic tests for the portable scientific interfaces."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from biological_age_prediction_and_mortality_analysis.analyze_external_disease_differences import (
    age_correct_and_test,
)
from biological_age_prediction_and_mortality_analysis.config import (
    ENRICHMENT,
    EXCLUDED_MODELS,
    FIGURE4_COMPONENTS,
)
from biological_age_prediction_and_mortality_analysis.enrichment.build_component_weight_rankings import (  # noqa: E501
    build_ranking,
    expected_components,
)
from biological_age_prediction_and_mortality_analysis.enrichment.select_top_bottom_cpg_foregrounds import (  # noqa: E501
    select_foregrounds,
)
from biological_age_prediction_and_mortality_analysis.generate_inchianti_oof_predictions import (
    assign_age_bin_folds,
)
from biological_age_prediction_and_mortality_analysis.modeling import (
    ALPHAS,
    CV_FOLDS,
    MAX_ITER,
    fit_scaled_elasticnet_cv,
    validate_component_features,
)


class ManifestAndModelTests(unittest.TestCase):
    """Check exact manifests and leakage-safe Elastic Net configuration."""

    def test_manifest_has_exact_exclusion_and_features(self) -> None:
        self.assertEqual(EXCLUDED_MODELS, {"dnamfitage"})
        self.assertEqual(len(FIGURE4_COMPONENTS), 17)
        self.assertEqual((len(ALPHAS), CV_FOLDS), (30, 5))
        self.assertEqual(len(ENRICHMENT["chronological_candidates"]), 16)
        self.assertEqual(len(ENRICHMENT["chronological_affine_components"]), 8)
        self.assertEqual(len(ENRICHMENT["biological_candidates"]), 19)
        self.assertEqual(len(ENRICHMENT["biological_affine_components"]), 17)
        self.assertGreaterEqual(MAX_ITER, 20_000)
        validate_component_features(list(FIGURE4_COMPONENTS))
        with self.assertRaises(ValueError):
            validate_component_features(list(reversed(FIGURE4_COMPONENTS)))

    def test_fold_local_scaled_grid_search(self) -> None:
        rng = np.random.default_rng(17)
        x = rng.normal(size=(60, 4)) * np.array([1.0, 10.0, 0.1, 3.0])
        y = 2.0 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(scale=0.2, size=60)
        fitted = fit_scaled_elasticnet_cv(x, y, n_jobs=1)
        self.assertEqual(list(fitted.model.named_steps), ["scaler", "elasticnet"])
        self.assertIn(fitted.alpha, ALPHAS)
        self.assertEqual(len(fitted.alpha_search), 30)
        self.assertLess(fitted.n_iter, MAX_ITER)

    def test_age_bin_fold_assignment_is_deterministic(self) -> None:
        age = pd.Series(np.repeat([30, 40, 50, 60, 70], 7))
        first, second = assign_age_bin_folds(age, 5, 0), assign_age_bin_folds(age, 5, 0)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first), set(range(5)))


class ExternalAndEnrichmentTests(unittest.TestCase):
    """Check age correction, coefficient aggregation, and foreground selection."""

    def test_external_age_correction_and_three_model_bonferroni(self) -> None:
        rows: list[dict[str, object]] = []
        for dataset in ("d1", "d2"):
            for model_index, model in enumerate(("bio_metaclock", "baseline", "dnamphenoage")):
                for group, shift in (("control", 0.0), ("disease", 1.5)):
                    for index, age in enumerate((40.0, 50.0, 60.0)):
                        rows.append(
                            {
                                "dataset": dataset,
                                "disease": f"{dataset}_disease",
                                "sample_id": f"{dataset}_{group}_{index}",
                                "age": age,
                                "group": group,
                                "model": model,
                                "pred_age": age * 0.5 + shift + model_index * 0.1 + index * 0.03,
                            }
                        )
        corrected, fits, tests = age_correct_and_test(pd.DataFrame(rows), expected_datasets=2)
        self.assertFalse(corrected["difference"].isna().any())
        self.assertEqual((len(fits), len(tests)), (6, 6))
        self.assertTrue(tests["bonferroni_p_value"].between(0, 1).all())

    def test_component_ranking_uses_raw_coefficient_products(self) -> None:
        components = expected_components("chrono")
        meta = pd.DataFrame(
            {
                "model": "chrono",
                "clock": components,
                "coefficient": np.arange(1, len(components) + 1),
            }
        )
        cpg = pd.DataFrame(
            [
                {"clock": clock, "cpg": cpg_name, "coefficient": coefficient}
                for clock in components
                for cpg_name, coefficient in (("cg1", 1.0), ("cg2", -0.5))
            ]
        )
        audit = pd.DataFrame(
            {"clock": components, "fixed_affine": True, "artifact_sha256": "a" * 64}
        )
        ranking, contributions = build_ranking("chrono", meta, cpg, audit)
        total = sum(range(1, len(components) + 1))
        self.assertEqual(
            dict(zip(ranking["cpg"], ranking["weight"])), {"cg1": total, "cg2": -0.5 * total}
        )
        self.assertEqual(len(contributions), 2 * len(components))

    def test_foregrounds_are_exact_deterministic_extremes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ranking_path = root / "ranking.csv"
            pd.DataFrame(
                {"cpg": [f"cg{i}" for i in range(6)], "weight": [0, 3, -2, 3, -5, 1]}
            ).to_csv(ranking_path, index=False)
            union = pd.DataFrame(
                {
                    "cpg": [f"cg{i}" for i in range(6)],
                    "chromosome": "chr1",
                    "position": np.arange(101, 107),
                    "in_27k": False,
                    "in_450k": True,
                    "in_epic": True,
                }
            )
            audit = select_foregrounds(ranking_path, union, root / "selected", top_n=2)
            top = pd.read_csv(root / "selected" / "top_selected_cpgs.csv")
            bottom = pd.read_csv(root / "selected" / "bottom_selected_cpgs.csv")
            self.assertEqual(top["cpg"].tolist(), ["cg1", "cg3"])
            self.assertEqual(bottom["cpg"].tolist(), ["cg4", "cg2"])
            self.assertEqual(audit["top"]["regions"], 2)


@unittest.skipUnless(importlib.util.find_spec("lifelines"), "lifelines is not installed")
class MortalityTests(unittest.TestCase):
    """Check formal KM schemas when the project survival dependency is installed."""

    def test_floor_tail_and_km_audit_schema(self) -> None:
        from biological_age_prediction_and_mortality_analysis.analyze_mortality_associations import (  # noqa: E501
            fit_tail_survival,
        )

        rows = [
            {
                "model": model,
                "sampleID": f"{model}_{index}",
                "age_corrected_prediction": index,
                "survival_time": 1 + index % 6,
                "event": int(index % 3 != 0),
            }
            for model in ("bio_metaclock", "baseline", "clinical_phenoage")
            for index in range(11)
        ]
        curves, tests = fit_tail_survival(pd.DataFrame(rows))
        self.assertTrue(
            {"time", "survival", "ci_lower", "ci_upper", "n_at_risk", "n_events"}.issubset(curves)
        )
        self.assertTrue(tests[["bottom_20_n", "top_20_n"]].eq(2).all().all())
        self.assertTrue(
            {"observed_top_events", "expected_top_events", "logrank_variance"}.issubset(tests)
        )


if __name__ == "__main__":
    unittest.main()
