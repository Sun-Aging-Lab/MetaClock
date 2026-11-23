"""Repository-local defaults shared by the intervention workflow."""

from __future__ import annotations

from hashlib import file_digest
from pathlib import Path

from repository_paths import PROJECT_ROOT, relative_path, repo_child, repo_path

__all__ = ["PROJECT_ROOT", "relative_path", "repo_child", "repo_path", "plot_input", "portable_reference", "file_sha256"]

DEFAULT_FETCH_DIR = PROJECT_ROOT / "data" / "intervention" / "geo_fetch_77"
DEFAULT_MATRIX_ROOT = DEFAULT_FETCH_DIR / "matrices"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "intervention" / "biometaclock_94"
DEFAULT_TRANSLAGE_DIR = PROJECT_ROOT / "data" / "intervention" / "translage"
DEFAULT_TRANSLAGE_MANIFEST = (
    DEFAULT_TRANSLAGE_DIR / "catalog" / "official_analysis_manifest.csv"
)
DEFAULT_TRANSLAGE_DATA_ROOT = DEFAULT_TRANSLAGE_DIR / "official"
DEFAULT_PYAGING_DIR = PROJECT_ROOT / "data" / "intervention" / "pyaging_cache"
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "biological_age_prediction_and_mortality_analysis" / "model_weights"
)


def plot_input(value: str) -> Path:
    """Resolve a supplied input file or an upstream repository-relative artifact."""
    return PROJECT_ROOT / Path(value).expanduser()


def portable_reference(path: Path) -> str:
    """Use a repository-relative reference or a replaceable external file path."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return relative_path(resolved)
    return (Path("path/to/your") / path.name).as_posix()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()
