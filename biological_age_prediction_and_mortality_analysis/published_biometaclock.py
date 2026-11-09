"""Export and load the 672-participant biological MetaClock formula."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from repository_paths import relative_path
from .provenance import software_versions


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model_weights"
MODEL_WEIGHTS_FILE = "biometaclock_672_participants_weights.csv"
MODEL_METADATA_FILE = "biometaclock_672_participants_metadata.json"


@dataclass(frozen=True)
class PublishedBioMetaClock:
    """Affine model reconstructed from the published repository weights."""

    features: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    weights_path: Path
    metadata_path: Path

    @property
    def n_features_in_(self) -> int:
        return len(self.features)

    @property
    def parameters(self) -> dict[str, object]:
        """Readable prediction parameters saved with reusable results."""
        return {
            "features": list(self.features),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
        }

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"BioMetaClock requires {self.n_features_in_} ordered components; "
                f"received shape {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("BioMetaClock component inputs must be finite")
        return self.intercept + values @ self.coefficients


def export_published_biometaclock(
    model,
    features: list[str],
    model_dir: Path,
    *,
    training_participants: int,
    search_config: dict[str, object],
    training_software: dict[str, str] | None = None,
    cv_convergence_warnings: int | None = None,
) -> tuple[Path, Path]:
    """Export a fitted zero-imputation/scaler/ElasticNet pipeline as affine weights."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_is_fitted

    if not isinstance(model, Pipeline) or len(model.steps) != 3:
        raise ValueError("Expected imputer, scaler, and ElasticNet pipeline steps.")
    imputer, scaler, estimator = (step for _, step in model.steps)
    if not (isinstance(imputer, SimpleImputer) and isinstance(scaler, StandardScaler)
            and isinstance(estimator, ElasticNet)):
        raise ValueError("Unsupported BioMetaClock pipeline architecture.")
    for step in (imputer, scaler, estimator):
        check_is_fitted(step)
    if (imputer.strategy != "constant" or imputer.fill_value != 0.0
            or imputer.add_indicator or not imputer.keep_empty_features
            or not scaler.with_mean or not scaler.with_std):
        raise ValueError("Affine export requires zero imputation and standard scaling.")
    features = list(features)
    if (not features or len(set(features)) != len(features)
            or len(features) != int(model.n_features_in_)
            or np.asarray(estimator.coef_).shape != (len(features),)):
        raise ValueError("Feature manifest must match all fitted input coefficients.")
    if training_participants <= 0 or not np.all(scaler.n_samples_seen_ == training_participants):
        raise ValueError("Participant count differs from the scaler's fitted sample count.")
    means = np.asarray(scaler.mean_, dtype=np.float64)
    scales = np.asarray(scaler.scale_, dtype=np.float64)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    if not np.isfinite(np.r_[means, scales, coefficients, estimator.intercept_]).all() or np.any(scales <= 0):
        raise ValueError("Fitted affine parameters must be finite, with positive scales.")
    raw_coefficients = coefficients / scales
    raw_intercept = float(estimator.intercept_ - means @ raw_coefficients)
    weights = pd.DataFrame({
        "feature": features, "scaler_mean": means, "scaler_scale": scales,
        "standardized_coefficient": coefficients, "raw_coefficient": raw_coefficients,
    })
    versions = software_versions(("scikit-learn", "numpy", "pandas", "joblib"))
    metadata = {
        "model": "BioMetaClock", "training_participants": int(training_participants),
        "features_file_order": MODEL_WEIGHTS_FILE, "features": features,
        "feature_count": len(features),
        "estimator": "SimpleImputer followed by StandardScaler and ElasticNet",
        "alpha": float(estimator.alpha), "l1_ratio": float(estimator.l1_ratio),
        "standardized_intercept": float(estimator.intercept_),
        "raw_intercept": raw_intercept, "n_iter": int(estimator.n_iter_),
        "max_iter": int(estimator.max_iter), "tol": float(estimator.tol),
        "selection": estimator.selection, "random_state": estimator.random_state,
        "dual_gap": float(estimator.dual_gap_),
        "selected_refit_at_iteration_cap": bool(estimator.n_iter_ >= estimator.max_iter),
        "cv_convergence_warnings": cv_convergence_warnings,
        "imputation": {"strategy": "constant", "fill_value": 0.0, "keep_empty_features": True},
        "scaling": {"with_mean": True, "with_std": True},
        "prediction_input": "finite ordered components; replace missing or nonfinite values with zero",
        "search_config": search_config,
        "coefficient_formula": "raw_coefficient = standardized_coefficient / scaler_scale",
        "prediction_formula": "raw_intercept + X @ raw_coefficient",
        "training_software": training_software if training_software is not None else versions,
        "extraction_software": versions,
        "restricted_data_note": "Restricted InCHIANTI participant data are not distributed.",
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / MODEL_WEIGHTS_FILE
    metadata_path = model_dir / MODEL_METADATA_FILE
    temporary_weights = weights_path.with_suffix(".csv.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    weights.to_csv(temporary_weights, index=False, float_format="%.17g")
    temporary_metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_weights.replace(weights_path)
    temporary_metadata.replace(metadata_path)
    return weights_path, metadata_path


def load_published_biometaclock(
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> tuple[PublishedBioMetaClock, list[str]]:
    """Load and cross-validate the tracked weights and metadata."""
    weights_path = model_dir / MODEL_WEIGHTS_FILE
    metadata_path = model_dir / MODEL_METADATA_FILE
    if not weights_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Missing published BioMetaClock files: "
            f"{relative_path(weights_path)} and {relative_path(metadata_path)}"
        )
    weights = pd.read_csv(weights_path, float_precision="round_trip")
    required = {
        "feature",
        "scaler_mean",
        "scaler_scale",
        "standardized_coefficient",
        "raw_coefficient",
    }
    if missing := required.difference(weights.columns):
        raise KeyError(f"BioMetaClock weights are missing columns: {sorted(missing)}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = weights["feature"].astype(str).tolist()
    if "features" in metadata and features != metadata["features"]:
        raise ValueError("BioMetaClock feature order differs from its metadata.")
    coefficients = pd.to_numeric(weights["raw_coefficient"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if len(features) != int(metadata["feature_count"]) or len(set(features)) != len(
        features
    ):
        raise ValueError("BioMetaClock feature manifest is incomplete or duplicated")
    if not np.isfinite(coefficients).all():
        raise ValueError("BioMetaClock coefficients must be finite")
    intercept = float(metadata["raw_intercept"])
    if not np.isfinite(intercept):
        raise ValueError("BioMetaClock intercept must be finite")
    means = pd.to_numeric(weights["scaler_mean"], errors="raise").to_numpy(
        dtype=np.float64
    )
    scales = pd.to_numeric(weights["scaler_scale"], errors="raise").to_numpy(
        dtype=np.float64
    )
    standardized = pd.to_numeric(
        weights["standardized_coefficient"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ValueError("BioMetaClock scaler parameters must be finite and positive")
    if not np.allclose(coefficients, standardized / scales, rtol=1e-12, atol=1e-14):
        raise ValueError("BioMetaClock raw and standardized coefficients disagree")
    reconstructed_intercept = float(
        metadata["standardized_intercept"] - means @ coefficients
    )
    if not np.isclose(intercept, reconstructed_intercept, rtol=1e-12, atol=1e-12):
        raise ValueError("BioMetaClock raw and standardized intercepts disagree")
    model = PublishedBioMetaClock(
        tuple(features), coefficients, intercept, weights_path, metadata_path
    )
    return model, features
