#!/usr/bin/env python3
"""Fit fixed-affine component clocks and rank CpGs for restricted-background GREAT."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ENRICHMENT, MANIFEST_PATH
from ..data import complete_case_mask, read_analysis_dataset
from ..production_oof import primary_samples
from ..provenance import software_versions, write_json
from ..run_biological_age_prediction import SEED, search_fold_local_elasticnet

TRAIN_DATASETS = (
    "GSE40279", "GSE51057", "GSE42861", "GSE41169", "GSE51032",
    "GSE73103", "GSE69270", "GSE64495", "GSE52588",
)
MODEL_NAMES = {"chrono": "chronological", "bio": "biological"}
ALPHAS = np.logspace(0, -5, 30)


def source_record(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        return {"path": str(path.resolve()), "sha256": hashlib.file_digest(source, "sha256").hexdigest()}


def prepare_training(
    model: str, predictions_path: Path, matched_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Select the model's training profiles and fixed-affine features."""
    features = list(ENRICHMENT[f"{MODEL_NAMES[model]}_affine_components"])
    if model == "chrono":
        outputs = pd.read_csv(predictions_path)
        frame = outputs.loc[outputs.dataset.isin(TRAIN_DATASETS)].copy()
        values = frame[features].apply(pd.to_numeric, errors="coerce")
        ages = pd.to_numeric(frame.age, errors="coerce")
        complete = ages.notna() & ~values.isna().any(axis=1)
        frame, values, ages = frame.loc[complete], values.loc[complete], ages.loc[complete]
        if set(frame.dataset) != set(TRAIN_DATASETS) or len(frame) != 3353:
            raise ValueError("Expected 3,353 complete profiles from the nine training datasets.")
        target = ages.to_numpy(dtype=np.float64)
    else:
        if matched_path is None:
            raise ValueError("Biological training requires --matched-samples.")
        matched = read_analysis_dataset(
            matched_path,
            columns=[
                "ID", "sampleID", "age_years", "sex_male", "clinical_phenoage",
                "clinical_phenoage_complete_case",
            ],
        )
        complete = complete_case_mask(matched["clinical_phenoage_complete_case"])
        matched = matched.loc[complete].dropna(
            subset=["ID", "sampleID", "age_years", "sex_male", "clinical_phenoage"]
        ).copy()
        primary = primary_samples(matched)[["ID", "sampleID"]]
        matched_target = matched[["sampleID", "clinical_phenoage"]]
        outputs = pd.read_csv(predictions_path, float_precision="round_trip")
        columns = ["sampleID", *features]
        if "clinical_phenoage" in outputs:
            columns.append("clinical_phenoage")
        frame = primary.merge(matched_target, on="sampleID", validate="one_to_one").merge(
            outputs[columns], on="sampleID", validate="one_to_one", suffixes=("", "_components"),
        )
        if not frame.sampleID.equals(primary.sampleID):
            raise ValueError("Missing metadata or component predictions for the 672 primary profiles.")
        target = pd.to_numeric(frame.clinical_phenoage, errors="raise").to_numpy(np.float64)
        if "clinical_phenoage_components" in frame:
            np.testing.assert_allclose(target, frame.clinical_phenoage_components, rtol=0, atol=1e-8)
        values = frame[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    matrix = values.to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (model == "chrono" and not np.isfinite(matrix).all()):
        raise ValueError("Nonfinite training target or chronological component prediction.")
    return frame.reset_index(drop=True), matrix, target, features


def fit_meta_model(
    matrix: np.ndarray, target: np.ndarray, model: str, n_jobs: int = 1,
) -> tuple[Pipeline, GridSearchCV]:
    """Select alpha with fold-local preprocessing and refit on all training profiles."""
    if model == "bio":
        search = search_fold_local_elasticnet(matrix, target, n_jobs=n_jobs)
        return search.best_estimator_, search
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            l1_ratio=0.5, max_iter=20000, selection="cyclic", precompute=True,
            tol=1e-4, random_state=42,
        )),
    ])
    search = GridSearchCV(
        pipeline, {"model__alpha": ALPHAS},
        cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_squared_error",
        n_jobs=n_jobs, refit=False, error_score="raise",
    ).fit(matrix, target)
    fitted = clone(pipeline).set_params(**search.best_params_, model__precompute=False)
    fitted.fit(matrix, target)
    return fitted, search


def aggregate_cpg_weights(
    coefficients_path: Path, features: list[str], raw_coefficients: np.ndarray, model: str,
) -> pd.DataFrame:
    """Compose audited component CpG coefficients with raw-scale meta weights."""
    rename = (
        {"clock_name": "clock", "cpg_site": "cpg", "clock_cpg_weight": "component_weight"}
        if model == "chrono" else {"clock_feature_weight": "component_weight"}
    )
    source = pd.read_csv(coefficients_path).rename(columns=rename)
    source = source[["clock", "cpg", "component_weight"]].copy()
    if set(source.clock) != set(features):
        raise ValueError("Audited CpG coefficient clocks differ from the component manifest.")
    source["component_weight"] = pd.to_numeric(source.component_weight, errors="raise")
    if source.cpg.isna().any() or not np.isfinite(source.component_weight).all():
        raise ValueError("CpG identifiers and component coefficients must be valid.")
    source["weight"] = source.component_weight * source.clock.map(dict(zip(features, raw_coefficients)))
    return source.groupby("cpg", as_index=False).agg(weight=("weight", "sum")).sort_values(
        ["weight", "cpg"], ascending=[False, True],
    ).reset_index(drop=True)


def select_cpgs(ranking: pd.DataFrame, membership_path: Path, model: str) -> pd.DataFrame:
    """Select extremes after restricting the full ranking to the array union."""
    union = pd.read_csv(membership_path, usecols=["cpg"])
    if len(union) != int(ENRICHMENT["union_cpgs"]) or union.cpg.duplicated().any():
        raise ValueError("Expected the complete unique 27K/450K/EPIC union.")
    eligible = ranking.loc[ranking.cpg.isin(union.cpg)]
    count = int(ENRICHMENT["top_n"])
    if len(eligible) < 2 * count:
        raise ValueError("Insufficient ranked CpGs in the array union.")
    selections = []
    for direction, ascending in (("top", False), ("bottom", True)):
        selected = eligible.sort_values(["weight", "cpg"], ascending=[ascending, True]).head(count).copy()
        selected.insert(0, "rank", range(1, count + 1))
        selected.insert(0, "direction", direction)
        selected.insert(0, "group", MODEL_NAMES[model])
        selections.append(selected)
    return pd.concat(selections, ignore_index=True)


def write_rankings(args: argparse.Namespace, weights: pd.DataFrame, weights_path: Path) -> None:
    features = ENRICHMENT[f"{MODEL_NAMES[args.model]}_affine_components"]
    if weights.clock.tolist() != features or not np.isfinite(weights.effective_raw_coefficient).all():
        raise ValueError("Meta weights must match the ordered fixed-affine component list.")
    ranking = aggregate_cpg_weights(
        args.component_weights, features, weights.effective_raw_coefficient.to_numpy(), args.model,
    )
    selected = select_cpgs(ranking, args.membership, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(args.output_dir / "meta_clock_weights.csv", index=False)
    ranking.to_csv(args.output_dir / "cpg_weights.csv.gz", index=False)
    selected.to_csv(args.output_dir / "selected_cpg_weights.csv", index=False)
    write_json(args.output_dir / "ranking_manifest.json", {
        "model": args.model, "weights": source_record(weights_path),
        "component_weights": source_record(args.component_weights), "membership": source_record(args.membership),
        "configuration": source_record(MANIFEST_PATH), "script": source_record(Path(__file__)),
        "software": software_versions(("numpy", "pandas")),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODEL_NAMES))
    parser.add_argument("--predictions", type=Path,
                        help="Refit from component predictions. By default reuse the bundled meta weights.")
    parser.add_argument(
        "--matched-samples", type=Path,
        help=("Authorized full prepared cohort with ID, sampleID, age_years, sex_male, "
              "clinical_phenoage, and clinical_phenoage_complete_case."),
    )
    parser.add_argument("--component-weights", type=Path, required=True,
                        help="Audited fixed-affine component CpG coefficient CSV/CSV.GZ.")
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for the CV search.")
    args = parser.parse_args()
    if args.predictions is None:
        if args.matched_samples:
            parser.error("--matched-samples requires --predictions.")
        if (args.output_dir / "model.joblib").exists():
            parser.error("Use a separate output directory for rankings from bundled weights.")
        weights_path = Path(__file__).with_name(f"{args.model}_meta_weights.csv")
        write_rankings(args, pd.read_csv(weights_path, float_precision="round_trip"), weights_path)
        return
    frame, matrix, target, features = prepare_training(args.model, args.predictions, args.matched_samples)
    fitted, search = fit_meta_model(matrix, target, args.model, args.n_jobs)
    scaler, estimator = fitted.named_steps["scaler"], fitted.steps[-1][1]
    raw = estimator.coef_ / scaler.scale_
    intercept = float(estimator.intercept_ - scaler.mean_ @ raw)
    predictions = fitted.predict(matrix)
    raw_matrix = fitted.named_steps["imputer"].transform(matrix) if args.model == "bio" else matrix
    np.testing.assert_allclose(predictions, intercept + raw_matrix @ raw, rtol=1e-12, atol=1e-10)
    weights = pd.DataFrame({
        "clock": features, "standardized_coefficient": estimator.coef_,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "effective_raw_coefficient": raw,
    })
    write_rankings(args, weights, args.output_dir / "meta_clock_weights.csv")
    joblib.dump(fitted, args.output_dir / "model.joblib")
    pd.DataFrame(search.cv_results_).to_csv(args.output_dir / "cv_results.csv", index=False)
    splits = search.cv.split(matrix) if isinstance(search.cv, KFold) else search.cv
    fold = np.empty(len(frame), dtype=int)
    for number, (_, validation) in enumerate(splits):
        fold[validation] = number
    keys = [name for name in ("dataset", "ID", "sampleID", "sample_id") if name in frame]
    frame[keys].assign(target=target, predicted=predictions, inner_validation_fold=fold).to_csv(
        args.output_dir / "training_predictions.csv", index=False,
    )
    sources = {"predictions": args.predictions, "component_weights": args.component_weights,
               "membership": args.membership, "script": Path(__file__), "configuration": MANIFEST_PATH}
    if args.matched_samples is not None:
        sources["matched_samples"] = args.matched_samples
    if args.model == "bio":
        sources["primary_split_configuration"] = MANIFEST_PATH
        sources["fit_implementation"] = MANIFEST_PATH.with_name("run_biological_age_prediction.py")
    write_json(args.output_dir / "training_manifest.json", {
        "model": args.model, "components": features, "samples": len(frame),
        "excluded_before_fit": ENRICHMENT[f"{MODEL_NAMES[args.model]}_non_affine_components"],
        "cv": {"class": "KFold", "n_splits": 5, "shuffle": True, "random_state": SEED if args.model == "bio" else 42},
        "scaling": "StandardScaler fitted separately within each inner training fold",
        "cv_precompute": args.model == "chrono", "refit_precompute": False,
        "imputation": "fold-local constant zero" if args.model == "bio" else None,
        "estimator": estimator.get_params(), "alphas": ALPHAS.tolist(),
        "best_mean_validation_mse": float(-search.best_score_),
        "n_iter": int(estimator.n_iter_), "dual_gap": float(estimator.dual_gap_),
        "raw_intercept": intercept, "selection_universe": "27K/450K/EPIC hg19 union",
        "sources": {name: source_record(path) for name, path in sources.items()},
        "software": software_versions(("numpy", "pandas", "scikit-learn", "joblib")),
    })


if __name__ == "__main__":
    main()
