"""Portable healthy-tissue dataset declarations."""

from pathlib import Path

from generalization.datasets import CHRONOLOGICAL_COMPONENTS
from generalization.modeling import DatasetSpec

DATASETS = {
    "GSE232891": DatasetSpec(
        "GSE232891",
        "Saliva",
        Path("/path/to/your/file"),  # fill in your path
        Path("/path/to/your/file"),  # fill in your path
        CHRONOLOGICAL_COMPONENTS,
    ),
    "GSE287331": DatasetSpec(
        "GSE287331",
        "Breast",
        Path("/path/to/your/file"),  # fill in your path
        Path("/path/to/your/file"),  # fill in your path
        CHRONOLOGICAL_COMPONENTS,
    ),
    "GSE175458": DatasetSpec(
        "GSE175458",
        "Lung",
        Path("/path/to/your/file"),  # fill in your path
        Path("/path/to/your/file"),  # fill in your path
        CHRONOLOGICAL_COMPONENTS,
    ),
}
DEFAULT_SHARDS = Path("results/figure3/generalization/tissue/shards")
DEFAULT_SUMMARIES = Path("results/figure3/generalization/tissue/summaries")
DEFAULT_FIGURES = Path("figures/figure3/generalization/tissue")
