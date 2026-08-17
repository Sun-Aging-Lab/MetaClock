"""Portable array-platform dataset declarations."""

from pathlib import Path

from generalization.datasets import CHRONOLOGICAL_COMPONENTS
from generalization.modeling import DatasetSpec

DATASETS = {
    "GSE234461": DatasetSpec(
        "GSE234461",
        "MethylChipEPIC v1.0",
        Path("/path/to/your/file"),  # fill in your path
        Path("/path/to/your/file"),  # fill in your path
        CHRONOLOGICAL_COMPONENTS,
    ),
    "GSE184221": DatasetSpec(
        "GSE184221",
        "MethylChip40",
        Path("/path/to/your/file"),  # fill in your path
        Path("/path/to/your/file"),  # fill in your path
        CHRONOLOGICAL_COMPONENTS,
    ),
}
DEFAULT_SHARDS = Path("results/figure3/generalization/array/shards")
DEFAULT_SUMMARIES = Path("results/figure3/generalization/array/summaries")
DEFAULT_FIGURES = Path("figures/figure3/generalization/array")
