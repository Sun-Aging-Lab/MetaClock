"""Small deterministic tests for the portable scientific interfaces."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from biological_age_prediction_and_mortality_analysis.analyze_external_disease_differences import (
    age_correct_and_test,
)
from biological_age_prediction_and_mortality_analysis.apply_biometaclock_weights import (
    load_frozen_weights,
    predict_biometaclock,
)
from biological_age_prediction_and_mortality_analysis.config import (
    ENRICHMENT,
    EXCLUDED_MODELS,
    EXTERNAL_COMPONENTS,
)
from biological_age_prediction_and_mortality_analysis.enrichment.build_component_weight_rankings import (  # noqa: E501
    build_ranking,
    expected_components,
)
from biological_age_prediction_and_mortality_analysis.enrichment.select_top_bottom_cpg_foregrounds import (  # noqa: E501
    select_foregrounds,
)


class ManifestAndWeightTests(unittest.TestCase):
    """Check the public manifest and frozen BioMetaClock representation."""

    def test_manifest_has_exact_exclusion_and_features(self) -> None:
        self.assertEqual(EXCLUDED_MODELS, {"dnamfitage"})
        self.assertEqual(len(EXTERNAL_COMPONENTS), 17)
        self.assertEqual(len(ENRICHMENT["chronological_candidates"]), 16)
        self.assertEqual(len(ENRICHMENT["chronological_affine_components"]), 8)
        self.assertEqual(len(ENRICHMENT["biological_candidates"]), 19)
        self.assertEqual(len(ENRICHMENT["biological_affine_components"]), 17)

    def test_frozen_weights_predict_on_raw_scale(self) -> None:
        weights, intercept = load_frozen_weights()
        rng = np.random.default_rng(17)
        components = pd.DataFrame(
            rng.normal(size=(11, len(weights))), columns=weights["feature"]
        )
        expected = intercept + components.to_numpy() @ weights["raw_coefficient"].to_numpy()
        np.testing.assert_allclose(predict_biometaclock(components), expected, rtol=0, atol=1e-12)


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

    def test_published_top_bottom_weight_table(self) -> None:
        path = (
            Path(__file__).parents[1] / "enrichment" / "top_bottom_1000_cpg_weights.csv"
        )
        table = pd.read_csv(path)
        self.assertEqual(
            set(table.columns), {"group", "direction", "rank", "cpg", "weight"}
        )
        counts = table.groupby(["group", "direction"]).size()
        self.assertEqual(
            counts.to_dict(),
            {
                ("biological", "bottom"): 1000,
                ("biological", "top"): 1000,
                ("chronological", "bottom"): 1000,
                ("chronological", "top"): 1000,
            },
        )
        self.assertFalse(table.duplicated(["group", "direction", "cpg"]).any())

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


if __name__ == "__main__":
    unittest.main()
