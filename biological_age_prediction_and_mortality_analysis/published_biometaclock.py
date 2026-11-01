"""Load the tracked 672-participant biological MetaClock formula."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from repository_paths import relative_path


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
    weights = pd.read_csv(weights_path)
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
