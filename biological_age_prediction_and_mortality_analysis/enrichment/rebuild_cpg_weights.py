#!/usr/bin/env python3
"""Fit affine-component MetaClocks and rank CpGs for restricted-background GREAT.

Component predictions and audited fixed-affine CpG coefficients are inputs.
The chronological model uses the nine Figure 2 training datasets; the biological
model uses all 716 matched profiles in their input order. Non-affine clocks are
excluded using the analysis manifest.
"""

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

from ..config import ENRICHMENT
from ..provenance import software_versions, write_json

TRAIN_DATASETS = (
    "GSE40279", "GSE51057", "GSE42861", "GSE41169", "GSE51032",
    "GSE73103", "GSE69270", "GSE64495", "GSE52588",
)
MODEL_NAMES = {"chrono": "chronological", "bio": "biological"}
ALPHAS = np.logspace(0, -5, 30)


def prepare_training(
    model: str, predictions_path: Path, matched_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Preserve source row order and the model-specific component manifest."""
    features = list(ENRICHMENT[f"{MODEL_NAMES[model]}_affine_components"])
    outputs = pd.read_csv(predictions_path)
    if model == "chrono":
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
        matched = pd.read_csv(matched_path, usecols=["sampleID", "clinical_phenoage"])
        if matched.sampleID.duplicated().any() or outputs.sampleID.duplicated().any():
            raise ValueError("Training tables require unique sampleID values.")
        frame = matched.merge(
            outputs[["sampleID", "clinical_phenoage", *features]], on="sampleID",
            how="inner", sort=False, validate="one_to_one", suffixes=("", "_components"),
        )
        if len(frame) != 716 or len(frame) != len(matched) or len(frame) != len(outputs):
            raise ValueError("Expected all 716 matched biological training profiles.")
        target = pd.to_numeric(frame.clinical_phenoage, errors="raise").to_numpy(np.float64)
        np.testing.assert_allclose(
            target, pd.to_numeric(frame.clinical_phenoage_components, errors="raise"),
            rtol=0, atol=1e-8,
        )
        values = frame[features].apply(pd.to_numeric, errors="raise")
    matrix = values.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Affine-component training inputs must be finite.")
    return frame.reset_index(drop=True), matrix, target, features


def fit_meta_model(
    matrix: np.ndarray, target: np.ndarray, model: str, n_jobs: int = 1,
) -> tuple[Pipeline, GridSearchCV]:
    """Search with fold-local scaling; use Gram matrices in CV and direct refit."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            l1_ratio=0.5, max_iter=20000, selection="cyclic", precompute=True,
            tol=1e-4 if model == "chrono" else 1e-5,
            random_state=42 if model == "chrono" else None,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODEL_NAMES))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--matched-samples", type=Path)
    parser.add_argument("--component-weights", type=Path, required=True,
                        help="Audited fixed-affine component CpG coefficient CSV/CSV.GZ.")
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    frame, matrix, target, features = prepare_training(args.model, args.predictions, args.matched_samples)
    fitted, search = fit_meta_model(matrix, target, args.model, args.n_jobs)
    scaler, estimator = fitted.named_steps["scaler"], fitted.named_steps["model"]
    raw = estimator.coef_ / scaler.scale_
    intercept = float(estimator.intercept_ - scaler.mean_ @ raw)
    predictions = fitted.predict(matrix)
    np.testing.assert_allclose(predictions, intercept + matrix @ raw, rtol=1e-12, atol=1e-10)
    weights = pd.DataFrame({
        "clock": features, "standardized_coefficient": estimator.coef_,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "effective_raw_coefficient": raw,
    })
    ranking = aggregate_cpg_weights(args.component_weights, features, raw, args.model)
    selected = select_cpgs(ranking, args.membership, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(args.output_dir / "meta_clock_weights.csv", index=False)
    ranking.to_csv(args.output_dir / "cpg_weights.csv.gz", index=False)
    selected.to_csv(args.output_dir / "selected_cpg_weights.csv", index=False)
    joblib.dump(fitted, args.output_dir / "model.joblib")
    pd.DataFrame(search.cv_results_).to_csv(args.output_dir / "cv_results.csv", index=False)
    keys = [name for name in ("dataset", "sampleID", "sample_id") if name in frame]
    frame[keys].assign(target=target, predicted=predictions).to_csv(
        args.output_dir / "training_predictions.csv", index=False,
    )
    sources = {"predictions": args.predictions, "component_weights": args.component_weights,
               "membership": args.membership, "script": Path(__file__)}
    if args.matched_samples is not None:
        sources["matched_samples"] = args.matched_samples
    write_json(args.output_dir / "training_manifest.json", {
        "model": args.model, "components": features, "samples": len(frame),
        "cv": {"class": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "scaling": "StandardScaler fitted separately within each inner training fold",
        "cv_precompute": True, "refit_precompute": False,
        "estimator": estimator.get_params(), "alphas": ALPHAS.tolist(),
        "best_mean_validation_mse": float(-search.best_score_),
        "n_iter": int(estimator.n_iter_), "dual_gap": float(estimator.dual_gap_),
        "raw_intercept": intercept, "selection_universe": "27K/450K/EPIC hg19 union",
        "sources": {name: {"path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in sources.items()},
        "software": software_versions(("numpy", "pandas", "scikit-learn", "joblib")),
    })


if __name__ == "__main__":
    main()
