"""Focused regression tests for the 68-comparison manuscript screen."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats

from intervention_analysis import manuscript_screen as screen


PAIRED = "two_sided_paired_t_test"
WELCH = "two_sided_welch_t_test"


def _summary_row(test: str, control: np.ndarray, experiment: np.ndarray) -> dict:
    row = {
        "test": test,
        "control_n_donors": len(control),
        "experiment_n_donors": len(experiment),
        "control_mean": control.mean(),
        "experiment_mean": experiment.mean(),
        "control_sd": control.std(ddof=1),
        "experiment_sd": experiment.std(ddof=1),
        "paired_difference_sd": np.nan,
    }
    if test == PAIRED:
        row["paired_difference_sd"] = (experiment - control).std(ddof=1)
    return row


class DonorComparisonTests(unittest.TestCase):
    def test_paired_and_unequal_welch_match_scipy_raw_observations(self) -> None:
        paired_control = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
        paired_experiment = np.array([3.0, 2.5, 7.0, 8.0, 10.0, 16.0])
        welch_control = np.array([0.5, 1.2, 2.0, 3.7, 5.1])
        welch_experiment = np.array([1.0, 2.4, 2.8, 4.9, 6.2, 7.5, 9.0])
        frame = pd.DataFrame(
            [_summary_row(PAIRED, paired_control, paired_experiment),
             _summary_row(WELCH, welch_control, welch_experiment)],
            index=["paired", "welch"],
        )

        observed = screen.compare_donors(frame)
        paired_expected = stats.ttest_rel(paired_experiment, paired_control)
        welch_expected = stats.ttest_ind(welch_experiment, welch_control, equal_var=False)
        for label, expected in (("paired", paired_expected), ("welch", welch_expected)):
            self.assertAlmostEqual(
                observed.loc[label, "test_statistic_experiment_minus_control"],
                expected.statistic,
                places=12,
            )
            self.assertAlmostEqual(observed.loc[label, "two_sided_p_value"], expected.pvalue, places=12)
        self.assertAlmostEqual(
            observed.loc["paired", "mean_diff_experiment_minus_control"],
            paired_experiment.mean() - paired_control.mean(),
        )
        self.assertEqual(observed.loc["paired", "n_paired_donors_analyzed"], len(paired_control))
        self.assertEqual(observed.loc["welch", "n_paired_donors_analyzed"], 0)

    def test_invalid_summary_conditions_are_rejected(self) -> None:
        control = np.array([1.0, 2.0, 4.0, 8.0])
        experiment = np.array([2.0, 3.0, 7.0, 9.0])
        valid = pd.DataFrame([_summary_row(PAIRED, control, experiment)])
        mutations = {
            "too_few_donors": ("control_n_donors", 3),
            "fractional_donors": ("experiment_n_donors", 4.5),
            "negative_sd": ("control_sd", -1.0),
            "paired_count_mismatch": ("experiment_n_donors", 5),
            "missing_paired_sd": ("paired_difference_sd", np.nan),
            "zero_paired_sd": ("paired_difference_sd", 0.0),
            "unknown_test": ("test", "student_t_test"),
            "nonfinite_mean": ("control_mean", np.inf),
        }
        for name, (column, value) in mutations.items():
            with self.subTest(name=name):
                bad = valid.astype({"control_n_donors": float, "experiment_n_donors": float})
                bad.loc[0, column] = value
                with self.assertRaises(ValueError):
                    screen.compare_donors(bad)

        welch = pd.DataFrame([_summary_row(WELCH, control, experiment)])
        welch.loc[0, "paired_difference_sd"] = 1.0
        with self.assertRaises(ValueError):
            screen.compare_donors(welch)


class CanonicalFamilyTests(unittest.TestCase):
    def test_shipped_family_has_manuscript_counts_and_named_discoveries(self) -> None:
        family, decisions = screen.load_family()
        self.assertEqual(family.index.name, "family_test_id")
        self.assertTrue(family.index.is_unique)
        self.assertEqual(len(family), 68)
        self.assertEqual(family.public_id.nunique(), 54)
        self.assertEqual(family.test.value_counts().to_dict(), {PAIRED: 34, WELCH: 34})
        self.assertEqual(
            decisions.decision.value_counts().to_dict(),
            {"included": 68, "missing_individual_age": 24,
             "unresolved_donor_pairs": 1, "newborn_age_category": 1},
        )

        results = screen.compare_donors(family)
        results["q"] = screen.bh_adjust(results.two_sided_p_value)[0]
        self.assertEqual(int(results.q.le(0.05).sum()), 4)
        self.assertEqual(int(results.q.le(0.20).sum()), 12)
        self.assertEqual(set(results.index[results.q.le(0.05)]), {
            "translage::GSE217633",           # Antiretroviral therapy
            "geo::curated_1fc35799c70a1f70",  # Abdominal radiotherapy
            "geo::curated_fe95adeead1b848b",  # Dexamethasone
            "geo::curated_a5aa77c6d7ffdccc",  # Cigarette smoking
        })

    def test_ledger_id_or_declared_test_mismatch_is_rejected(self) -> None:
        included = [
            {"family_test_id": f"included-{i}", "public_id": f"GSE{i % 54}", "decision": "included"}
            for i in range(68)
        ]
        eligibility = pd.DataFrame(
            included
            + [{"family_test_id": f"age-{i}", "public_id": f"AGE{i}",
                "decision": "missing_individual_age"} for i in range(24)]
            + [{"family_test_id": "pairs", "public_id": "PAIR",
                "decision": "unresolved_donor_pairs"},
               {"family_test_id": "newborn", "public_id": "GSE144804",
                "decision": "newborn_age_category"}]
        )
        statistics = pd.DataFrame([
            {"family_test_id": row["family_test_id"], "public_id": row["public_id"],
             "analysis_method": "donor_level_welch", "test": WELCH}
            for row in included
        ])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            eligibility_path, statistics_path = root / "eligibility.csv", root / "statistics.csv"
            eligibility.to_csv(eligibility_path, index=False)
            statistics.loc[0, "family_test_id"] = "wrong-id"
            statistics.to_csv(statistics_path, index=False)
            with patch.object(screen, "ELIGIBILITY", eligibility_path), patch.object(
                screen, "DONOR_STATISTICS", statistics_path
            ):
                with self.assertRaisesRegex(ValueError, "Comparison IDs or public accessions"):
                    screen.load_family()

            statistics.loc[0, "family_test_id"] = included[0]["family_test_id"]
            statistics.loc[0, "test"] = PAIRED
            statistics.to_csv(statistics_path, index=False)
            with patch.object(screen, "ELIGIBILITY", eligibility_path), patch.object(
                screen, "DONOR_STATISTICS", statistics_path
            ):
                with self.assertRaisesRegex(ValueError, "declared test"):
                    screen.load_family()


if __name__ == "__main__":
    unittest.main()
