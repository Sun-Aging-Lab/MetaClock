"""Reproduce the external MetaClock-19 donor comparisons reported in Figure 5."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy import stats

from intervention_analysis.paths import file_sha256, portable_reference
from intervention_analysis.run_external_candidate_screen import bh_adjust

CONFIG = Path(__file__).resolve().parent / "config"
ELIGIBILITY = CONFIG / "external_family_eligibility.csv"
DONOR_STATISTICS = CONFIG.parent / "screen_summaries" / "external_metaclock19.csv"
RESULT_NAME = "combined_geo_translage_results_bh.csv"
DESIGNS = {
    "donor_level_paired": "two_sided_paired_t_test",
    "donor_level_partial_paired": "two_sided_paired_t_test",
    "donor_level_welch": "two_sided_welch_t_test",
    "donor_pre_post_change_welch": "two_sided_welch_t_test",
}
MOMENTS = ["control_n_donors", "experiment_n_donors", "control_mean", "experiment_mean",
           "control_sd", "experiment_sd", "paired_difference_sd"]


def load_family() -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = pd.read_csv(ELIGIBILITY).set_index("family_test_id", verify_integrity=True)
    expected = {"included": 68, "missing_individual_age": 24,
                "unresolved_donor_pairs": 1, "newborn_age_category": 1}
    if decisions.decision.value_counts(dropna=False).to_dict() != expected:
        raise ValueError("Unexpected composition of the reviewed external screen.")
    admitted = decisions.loc[decisions.decision.eq("included"), "public_id"].sort_index()
    family = pd.read_csv(DONOR_STATISTICS, float_precision="round_trip").set_index(
        "family_test_id", verify_integrity=True
    )
    if not family.public_id.sort_index().equals(admitted) or admitted.nunique() != 54:
        raise ValueError("Comparison IDs or public accessions differ from the eligibility ledger.")
    declared = family.analysis_method.map(DESIGNS)
    if declared.isna().any() or not declared.equals(family.test):
        raise ValueError("A donor design does not match its declared test.")
    return family, decisions


def compare_donors(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute experiment-minus-control effects and paired or Welch two-sided P values."""
    result = frame.drop(columns=["two_sided_p_value", "combined_bh_q_value",
                                 "mean_diff_experiment_minus_control"], errors="ignore").copy()
    finite = result[MOMENTS[:-1]].to_numpy(float)
    counts = result[["experiment_n_donors", "control_n_donors"]].to_numpy(float)
    if (not np.isfinite(finite).all() or (counts < 4).any() or (counts != np.floor(counts)).any()
            or (result[["experiment_sd", "control_sd"]] < 0).any().any()):
        raise ValueError("Invalid donor counts, means, or standard deviations.")
    paired = result.test.eq("two_sided_paired_t_test")
    if (not result.test.isin(DESIGNS.values()).all()
            or not np.isfinite(result.loc[paired, "paired_difference_sd"]).all()
            or not result.loc[paired, "paired_difference_sd"].gt(0).all()
            or not result.loc[~paired, "paired_difference_sd"].isna().all()
            or not result.loc[paired, "control_n_donors"].eq(result.loc[paired, "experiment_n_donors"]).all()):
        raise ValueError("Invalid paired differences or test assignment.")
    effect = result.experiment_mean - result.control_mean
    paired_n = result.loc[paired, "control_n_donors"]
    # The aggregate release has the sample SD of within-pair differences. SciPy's
    # paired t-test accepts observations only; evaluate the same Student t pivot here.
    paired_t = effect[paired] / (result.loc[paired, "paired_difference_sd"] / np.sqrt(paired_n))
    result.loc[paired, "test_statistic_experiment_minus_control"] = paired_t
    result.loc[paired, "two_sided_p_value"] = 2 * stats.t.sf(paired_t.abs().to_numpy(), df=paired_n.to_numpy() - 1)
    groups = result.loc[~paired]
    welch = stats.ttest_ind_from_stats(
        groups.experiment_mean.to_numpy(), groups.experiment_sd.to_numpy(), groups.experiment_n_donors.to_numpy(),
        groups.control_mean.to_numpy(), groups.control_sd.to_numpy(), groups.control_n_donors.to_numpy(),
        equal_var=False,
    )
    result.loc[~paired, "test_statistic_experiment_minus_control"] = welch.statistic
    result.loc[~paired, "two_sided_p_value"] = welch.pvalue
    if not np.isfinite(result[["test_statistic_experiment_minus_control", "two_sided_p_value"]]).all().all():
        raise ValueError("A donor contrast produced an undefined test.")
    result["mean_diff_experiment_minus_control"] = effect
    result["n_paired_donors_analyzed"] = result.control_n_donors.where(paired, 0)
    return result


def write_external_results(output: Path, run_dir: Path | None = None) -> None:
    panel, decisions = load_family()
    input_level = "donor summary statistics"
    model_inputs = {}
    if run_dir is not None:
        from intervention_analysis.external_scoring_bridge import (
            DEFAULT_WEIGHT_DIR, METADATA_FILE, WEIGHT_FILE, TASK_MANIFESTS, compute_moments,
        )

        raw = compute_moments(panel.reset_index(), run_dir).set_index("family_test_id", verify_integrity=True)
        if set(raw.index) != set(panel.index):
            raise ValueError("The prediction run did not cover exactly the reviewed comparisons.")
        panel[MOMENTS] = raw.loc[panel.index, MOMENTS]
        input_level = "19 component predictions and sample designs"
        model_inputs = {
            "weights_sha256": file_sha256(DEFAULT_WEIGHT_DIR / WEIGHT_FILE),
            "metadata_sha256": file_sha256(DEFAULT_WEIGHT_DIR / METADATA_FILE),
            "task_manifest_sha256": {name: file_sha256(run_dir / name) for name in TASK_MANIFESTS},
        }
    result = compare_donors(panel)
    result["combined_bh_q_value"], error = bh_adjust(result.two_sided_p_value)
    result["bh_q_value"] = result.combined_bh_q_value
    result["gse_id"] = result.public_id
    result["bh_family_n"] = len(result)
    result["model"] = "MetaClock19"
    result["n_component_features"] = 19
    result["age_adjustment"] = "within_gse_stratum_prediction_scale"
    result = result.reset_index().sort_values(["combined_bh_q_value", "two_sided_p_value", "family_test_id"])
    result["combined_rank"] = np.arange(1, len(result) + 1)
    q = result.combined_bh_q_value
    summary = {
        "model": "MetaClock19", "input_level": input_level,
        "candidate_comparisons": len(decisions), "eligibility": decisions.decision.value_counts().to_dict(),
        "n_interventions": len(result), "n_public_accessions": result.public_id.nunique(),
        "test_counts": result.test.value_counts().to_dict(),
        "bh_q_le_0_05": int(q.le(0.05).sum()), "bh_q_le_0_20": int(q.le(0.20).sum()),
        "effect": "experiment minus control, years", "bh_method": "Benjamini-Hochberg across all eligible comparisons",
        "bh_max_absolute_crosscheck_error": error, "eligibility_sha256": file_sha256(ELIGIBILITY),
        "donor_statistics_sha256": file_sha256(DONOR_STATISTICS) if run_dir is None else None,
        "prediction_run": portable_reference(run_dir) if run_dir else None,
        **model_inputs,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    results_csv = output / RESULT_NAME
    result.to_csv(results_csv, index=False)
    summary["results_sha256"] = file_sha256(results_csv)
    (output / "external_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
