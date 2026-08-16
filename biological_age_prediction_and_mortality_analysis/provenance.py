"""Small, reusable provenance helpers for normalized analysis inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd

try:
    from .config import FIGURE4, FIGURE4_COMPONENTS
except ImportError:  # Direct-script/import fallback from this directory.
    from config import FIGURE4, FIGURE4_COMPONENTS


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Describe one immutable workflow input."""
    resolved = path.resolve(strict=True)
    return {"path": str(path), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def software_versions(packages: Iterable[str]) -> dict[str, str]:
    """Record Python and requested installed distribution versions."""
    versions = {"python": platform.python_version()}
    for package in packages:
        versions[package] = importlib.metadata.version(package)
    return versions


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_all672_model_directory(model_dir: Path) -> dict[str, Any]:
    """Validate training metadata, exact features, fitted dimensions, and artifact hashes."""
    summary_path = model_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_n = int(FIGURE4["participants"])
    if (
        summary.get("refit_on_all_participants") is not True
        or summary.get("n_participants") != expected_n
    ):
        raise ValueError("External models were not refit on all 672 participants.")
    expected = {
        "bio_metaclock": len(FIGURE4_COMPONENTS),
        "baseline": int(FIGURE4["baseline_cpg_features"]),
    }
    for name, n_features in expected.items():
        metadata = summary.get("models", {}).get(name, {})
        if metadata.get("n_train") != expected_n or metadata.get("n_holdout") != 0:
            raise ValueError(f"{name} does not have the required all-672 provenance.")
        if metadata.get("n_features") != n_features or metadata.get("cv_convergence_warnings") != 0:
            raise ValueError(f"{name} feature count or convergence audit is invalid.")
        features = pd.read_csv(model_dir / f"{name}_features.csv")["feature"].astype(str).tolist()
        if name == "bio_metaclock" and features != list(FIGURE4_COMPONENTS):
            raise ValueError(
                "BioMeta feature manifest does not exactly match the canonical 17 clocks."
            )
        if len(features) != n_features or len(set(features)) != n_features:
            raise ValueError(f"{name} feature manifest is incomplete or duplicated.")
        feature_path = model_dir / f"{name}_features.csv"
        feature_hash = summary.get("feature_manifests", {}).get(name, {}).get("sha256")
        if not feature_hash or sha256(feature_path) != feature_hash:
            raise ValueError(f"{name} feature-manifest hash is missing or inconsistent.")
        artifact = model_dir / f"{name}.joblib"
        trained = joblib.load(artifact)
        if int(getattr(trained, "n_features_in_", -1)) != n_features:
            raise ValueError(f"{name} fitted artifact has an unexpected input dimension.")
        expected_hash = summary.get("artifacts", {}).get(name, {}).get("sha256")
        if expected_hash and sha256(artifact) != expected_hash:
            raise ValueError(f"{name} artifact hash does not match training_summary.json.")
    samples_path = model_dir / "training_samples.csv"
    samples = pd.read_csv(samples_path)
    if (
        len(samples) != expected_n
        or samples["ID"].nunique() != expected_n
        or samples["sampleID"].nunique() != expected_n
    ):
        raise ValueError("All-participant sample provenance is not exactly 672 unique IDs/samples.")
    samples_hash = summary.get("derived_outputs", {}).get("training_samples", {}).get("sha256")
    if not samples_hash or sha256(samples_path) != samples_hash:
        raise ValueError("All-participant sample provenance hash is missing or inconsistent.")
    return summary
