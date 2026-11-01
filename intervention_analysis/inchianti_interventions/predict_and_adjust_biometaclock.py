#!/usr/bin/env python3
"""Generate participant OOF BioMeta predictions and shared age/sex residuals.

Inputs are the existing matched-sample CSV and cached component-clock CSV.
The former must contain ID, sampleID, age_years, SEX (1=male, 2=female),
clinical_phenoage, and clinical_phenoage_complete_case. The latter must contain
sampleID and the 19 clocks ordered in the repository's biological manifest.
These clocks are already computed; this script does not process raw methylation.

The canonical Figure 4 helpers select one primary sample per participant and
map all samples from each participant to the same fixed outer fold. Each model
fits StandardScaler + ElasticNetCV on primary samples in the other four folds.
After prediction, one OLS model, pred_age ~ 1 + age_years + female, is fitted on
all 672 primary participants. Its residuals are shared by every intervention
comparison; they are not fitted separately within intervention groups.

Required CLI options: --matched-samples, --components, and --output-dir.
--fold-assignment defaults to the versioned Figure 4 assignment. Outputs include
participant_predictions_and_age_sex_residuals.csv (the next script's input),
all-sample OOF predictions, five fitted pipelines, and a provenance JSON with
input paths, feature order, model parameters, warnings, and software versions.
Only standard estimators are used; custom code handles alignment and reporting.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
import warnings

sys.dont_write_bytecode = True
# Support both direct execution in the repository and an installed/PYTHONPATH copy.
REPOSITORY = Path(__file__).resolve().parents[2]
if (REPOSITORY / "biological_age_prediction_and_mortality_analysis").is_dir():
    sys.path.insert(0, str(REPOSITORY))

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from biological_age_prediction_and_mortality_analysis import config, data


def prepare_samples(
    matched_path: Path, component_path: Path, assignment_path: Path
) -> pd.DataFrame:
    """Align complete-case samples, canonical components, and donor-level folds.

    Preserve stable sampleID order because ElasticNetCV's inner folds depend on
    row order. Participant selection follows the canonical lowest-numeric-suffix
    rule; repeated samples remain available for held-out prediction only.
    """
    columns = ["ID", "sampleID", "age_years", "SEX", "clinical_phenoage"]
    samples = pd.read_csv(
        matched_path, usecols=[*columns, "clinical_phenoage_complete_case"]
    )
    samples = samples.loc[
        data.complete_case_mask(samples.clinical_phenoage_complete_case), columns
    ].sort_values("sampleID", kind="mergesort").reset_index(drop=True)
    if (
        len(samples) != config.INCHIANTI_COMPLETE_CASE_SAMPLES
        or samples.ID.nunique() != config.INCHIANTI_PARTICIPANTS
        or samples.sampleID.duplicated().any()
        or samples[columns].isna().any().any()
    ):
        raise ValueError("Expected 716 complete-case samples from 672 unique participants.")
    if not np.isfinite(samples[["ID", "age_years", "clinical_phenoage"]]).all().all():
        raise ValueError("Participant IDs, ages, and target values must be finite.")
    if not samples.SEX.isin([1, 2]).all():
        raise ValueError("SEX must use the baseline codes 1=male and 2=female.")
    if samples.groupby("ID")[["SEX", "age_years", "clinical_phenoage"]].nunique().gt(1).any().any():
        raise ValueError("Repeated samples have inconsistent participant covariates.")

    features = list(config.OOF_COMPONENTS)
    header = pd.read_csv(component_path, nrows=0).columns
    # The optional cached target provides an additional alignment check.
    target_columns = ["clinical_phenoage"] if "clinical_phenoage" in header else []
    components = pd.read_csv(component_path, usecols=["sampleID", *features, *target_columns])
    if components.sampleID.isna().any() or components.sampleID.duplicated().any():
        raise ValueError("Component sampleID values must be present and unique.")
    samples = samples.merge(
        components, on="sampleID", how="left", validate="one_to_one",
        suffixes=("", "_components"), indicator=True, sort=False,
    )
    if not samples._merge.eq("both").all() or samples[features].isna().all(axis=1).any():
        raise ValueError("Component-clock records are missing for matched samples.")
    samples = samples.drop(columns="_merge")
    if target_columns:
        np.testing.assert_allclose(
            samples.clinical_phenoage, samples.clinical_phenoage_components,
            rtol=0, atol=1e-10, err_msg="Cached and matched Clinical PhenoAge disagree.",
        )
        samples = samples.drop(columns="clinical_phenoage_components")
    primary = data.select_one_sample_per_participant(samples)
    samples["selected_primary_sample"] = samples.sampleID.isin(primary.sampleID)
    samples["fold"] = data.fixed_figure4_oof_folds(samples, path=assignment_path)
    samples["SEX"] = samples.SEX.astype(int)
    return samples


def predict_oof(
    samples: pd.DataFrame, model_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit canonical five-fold models and predict every sample out of fold.

    Training includes one primary sample per donor. Every sample of a held-out
    donor is excluded from training, so repeated samples cannot leak across folds.
    Convergence warnings are recorded without changing the published estimator.
    """
    from biological_age_prediction_and_mortality_analysis import run_biological_age_prediction as training

    features = list(config.OOF_COMPONENTS)
    matrix = data.numeric_matrix(samples, features, dtype=np.float64)
    target = samples.clinical_phenoage.to_numpy(dtype=float)
    primary = samples.selected_primary_sample.to_numpy(dtype=bool)
    folds = samples.fold.to_numpy(dtype=int)
    predictions = np.full(len(samples), np.nan)
    fold_records: list[dict[str, Any]] = []
    model_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(config.INCHIANTI_OOF_FOLDS):
        train = primary & (folds != fold)
        held_out = folds == fold
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = training.fit_oof_elasticnet(matrix[train], target[train], seed=fold)
        predictions[held_out] = model.predict(matrix[held_out])
        model_path = model_dir / f"biometaclock_fold{fold}.joblib"
        joblib.dump(model, model_path)
        estimator = model.named_steps["elasticnetcv"]
        fold_records.append({
            "fold": fold, "n_train_participants": int(train.sum()),
            "n_test_participants": int((held_out & primary).sum()),
            "n_test_samples": int(held_out.sum()), "selected_alpha": float(estimator.alpha_),
            "final_fit_iterations": int(np.max(np.atleast_1d(estimator.n_iter_))),
            "convergence_warning_count": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
            "model_path": str(model_path.resolve()),
        })
        print(f"Fold {fold}: {int(train.sum())} training participants, "
              f"{int((held_out & primary).sum())} held-out participants", flush=True)
    if not np.isfinite(predictions).all():
        raise ValueError("Every sample must have a finite out-of-fold prediction.")

    output = samples[["ID", "sampleID", "age_years", "SEX", "clinical_phenoage",
                      "fold", "selected_primary_sample"]].copy()
    output["pred_age"] = predictions
    output["model"] = "bio_metaclock"
    output["prediction_scope"] = "participant_out_of_fold"
    output["n_component_features"] = len(features)
    estimator = model.named_steps["elasticnetcv"]
    parameters = {key: estimator.get_params()[key] for key in
                  ["l1_ratio", "cv", "max_iter", "tol", "selection", "n_jobs"]}
    parameters.update({
        "alphas": np.asarray(training.ALPHAS).tolist(),
        "random_state": "outer fold label 0..4", "scaling": "StandardScaler",
        "imputation": "non-finite component values replaced with zero",
        "training_order": "sampleID ascending; stable mergesort",
    })
    return output, {"parameters": parameters, "folds": fold_records,
                    "training_helper": str(Path(training.__file__).resolve())}


def adjust_age_sex(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one cohort-wide OLS and return fitted values and residuals in years.

    The input contains exactly one OOF prediction per primary participant. SEX
    is categorical: male is the reference and female is the indicator SEX == 2.
    """
    import statsmodels.api as sm

    people = predictions.sort_values("ID", kind="mergesort").reset_index(drop=True).copy()
    if (len(people) != config.INCHIANTI_PARTICIPANTS or people.ID.duplicated().any()
            or people.sampleID.duplicated().any()):
        raise ValueError("Age/sex correction requires one row per 672 participants.")
    if people[["ID", "sampleID"]].isna().any().any() or not people.SEX.isin([1, 2]).all():
        raise ValueError("Participant keys and baseline sex codes must be valid.")
    if not np.isfinite(people[["age_years", "pred_age"]]).all().all():
        raise ValueError("Age and prediction must be finite for the entire cohort.")
    design = sm.add_constant(
        people[["age_years"]].assign(female=people.SEX.eq(2).astype(int)),
        has_constant="add",
    )
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("Age/sex design matrix must have full column rank.")
    fit = sm.OLS(people.pred_age, design, missing="raise").fit()
    people["age_sex_fitted_bioage"] = fit.fittedvalues
    people["bioage_age_sex_residual"] = fit.resid
    np.testing.assert_allclose(design.T @ fit.resid.to_numpy(), 0, atol=1e-7)
    correction = {
        "formula": "pred_age ~ 1 + age_years + female",
        "equivalent_formula": "pred_age ~ age_years + C(SEX)",
        "sex_codes": {"1": "male reference", "2": "female"},
        "fit_n": int(fit.nobs), "single_cohort_wide_fit": True,
        "coefficients": fit.params.to_dict(), "r_squared": float(fit.rsquared),
        "residual_column": "bioage_age_sex_residual", "residual_units": "years",
    }
    return people, correction


def parse_args() -> argparse.Namespace:
    """Parse explicit input/output paths; do not create files during import."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matched-samples", type=Path, required=True,
                        help="Matched-sample CSV with clinical target, age, and SEX.")
    parser.add_argument("--components", type=Path, required=True,
                        help="Cached per-sample component clocks, keyed by sampleID.")
    parser.add_argument("--fold-assignment", type=Path,
                        default=config.INCHIANTI_OOF_FOLD_ASSIGNMENT_PATH,
                        help="Versioned donor-level CSV with columns ID,fold.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for predictions, fitted models, and provenance.")
    return parser.parse_args()


def main() -> None:
    """Run prediction followed by shared correction and save reproducibility data."""
    args = parse_args()
    started = datetime.now(timezone.utc).isoformat()
    paths = {"matched_samples": args.matched_samples, "component_clocks": args.components,
             "fold_assignment": args.fold_assignment, "analysis_manifest": config.MANIFEST_PATH,
             "data_helper": Path(data.__file__), "script": Path(__file__)}
    inputs = {name: str(path.resolve()) for name, path in paths.items()}
    samples = prepare_samples(args.matched_samples, args.components, args.fold_assignment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / "matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir.resolve() / "cache"))
    predictions, training = predict_oof(samples, args.output_dir / "fold_models")
    people, correction = adjust_age_sex(predictions.loc[predictions.selected_primary_sample])
    all_path = args.output_dir / "biometaclock_oof_predictions_all_samples.csv"
    people_path = args.output_dir / "participant_predictions_and_age_sex_residuals.csv"
    correction_path = args.output_dir / "age_sex_correction.json"
    predictions.to_csv(all_path, index=False)
    people.to_csv(people_path, index=False)
    correction_path.write_text(json.dumps(correction, indent=2, allow_nan=False) + "\n")
    manifest = {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "reference_data": "InCHIANTI Baseline_V8; Figure 4 primary DNAm cohort",
        "target": "clinical_phenoage", "features": list(config.OOF_COMPONENTS),
        "prediction_scope": "fixed five-fold participant out of fold",
        "counts": {"samples": len(predictions), "participants": len(people)},
        "primary_sample_rule": "lowest numeric sampleID suffix per participant",
        "training": training, "age_sex_correction": correction, "inputs": inputs,
        "outputs": [str(path.resolve()) for path in [all_path, people_path, correction_path]],
        "software": {"python": sys.version, "executable": sys.executable,
                     **{name: version(name) for name in
                        ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "joblib", "matplotlib"]}},
        "runtime_environment": {"platform": platform.platform(),
                                "threads": {key: os.environ.get(key) for key in
                                            ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}},
    }
    (args.output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved {len(people)} participant predictions and age/sex residuals: {people_path.resolve()}")


if __name__ == "__main__":
    main()
