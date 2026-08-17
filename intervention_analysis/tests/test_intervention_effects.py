"""Synthetic regression tests for intervention donor-design safeguards."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from intervention_analysis.biometaclock_modeling import (
    ALPHAS,
    fit_scaled_elastic_net_cv,
)
from intervention_analysis.select_curated_intervention_results import selected_mask
from intervention_analysis.summarize_intervention_effects import summarize


def comparison_rows(
    control_ids: list[str],
    experiment_ids: list[str],
    *,
    include_age: bool = False,
) -> pd.DataFrame:
    """Build one minimal synthetic intervention comparison."""
    rows: list[dict[str, object]] = []
    for group, donor_ids, offset in (
        ("control", control_ids, 0.0),
        ("experiment", experiment_ids, -2.0),
    ):
        for index, donor_id in enumerate(donor_ids):
            row: dict[str, object] = {
                "comparison_id": "cmp_1",
                "gse_id": "GSE1",
                "comparison_name": "synthetic",
                "sample_id": f"{group}_{donor_id}",
                "donor_id": donor_id,
                "group": group,
                "predicted_age": 50.0
                + index
                + offset
                - (0.1 * index if group == "experiment" else 0.0),
            }
            if include_age:
                row["chronological_age"] = 45.0 + index
            rows.append(row)
    return pd.DataFrame(rows)


class InterventionDesignTests(unittest.TestCase):
    """Exercise paired/unpaired and metadata validation without real data."""

    def test_no_age_fully_paired_design_is_accepted(self) -> None:
        results, _ = summarize(comparison_rows(["a", "b", "c", "d"], ["a", "b", "c", "d"]))
        self.assertEqual(results.at[0, "test"], "two-sided paired t-test")
        self.assertEqual(
            results.at[0, "age_adjustment"],
            "none_fully_paired_no_chronological_age",
        )

    def test_no_age_unpaired_design_is_rejected(self) -> None:
        frame = comparison_rows(["a", "b", "c", "d"], ["e", "f", "g", "h"])
        with self.assertRaisesRegex(ValueError, "without chronological age"):
            summarize(frame)

    def test_partially_paired_design_uses_shared_donors(self) -> None:
        frame = comparison_rows(
            ["a", "b", "c", "d", "e"],
            ["a", "b", "c", "d", "f"],
            include_age=True,
        )
        results, _ = summarize(frame)
        self.assertEqual(results.at[0, "test"], "two-sided paired t-test")
        self.assertEqual(results.at[0, "control_n_donors"], 4)
        self.assertEqual(results.at[0, "experiment_n_donors"], 4)

    def test_repeated_sample_with_conflicting_prediction_is_rejected(self) -> None:
        frame = comparison_rows(["a", "b", "c", "d"], ["e", "f", "g", "h"], include_age=True)
        duplicate = frame.iloc[[0]].copy()
        duplicate.loc[:, "predicted_age"] = 99.0
        with self.assertRaisesRegex(ValueError, "disagree"):
            summarize(pd.concat([frame, duplicate], ignore_index=True))

    def test_comparison_id_requires_constant_metadata(self) -> None:
        frame = comparison_rows(["a", "b"], ["c", "d"], include_age=True)
        frame.loc[frame.index[0], "gse_id"] = "GSE2"
        with self.assertRaisesRegex(ValueError, "one gse_id"):
            summarize(frame)

    def test_selected_mask_rejects_non_boolean_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "true/false"):
            selected_mask(pd.Series(["yes", "false"]))

    def test_canonical_scaled_alpha_search_is_auditable(self) -> None:
        rng = np.random.default_rng(9)
        matrix = rng.normal(size=(30, 3))
        target = 2.0 * matrix[:, 0] - matrix[:, 1] + rng.normal(0.0, 0.05, 30)
        fitted = fit_scaled_elastic_net_cv(matrix, target, n_jobs=1)
        self.assertEqual(len(fitted.alpha_search), len(ALPHAS))
        self.assertIn("standardscaler", fitted.model.named_steps)
        self.assertIn("elasticnet", fitted.model.named_steps)
        self.assertLess(fitted.n_iter, 100_000)
        self.assertEqual(fitted.cv_convergence_warnings, 0)
        self.assertEqual(fitted.refit_convergence_warnings, 0)
