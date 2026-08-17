"""Species names and replaceable relative input locations for Figure 3."""

from __future__ import annotations

from pathlib import Path

SPECIES = (
    "BosTaurus",
    "DelphinapterusLeucas",
    "EquusCaballus",
    "HeterocephalusGlaber",
    "HomoSapiens",
    "MacacaMulatta",
    "MusMusculus",
    "OvisAries",
    "PapioHamadryas",
    "RattusNorvegicus",
)
DISPLAY_NAMES = {
    "BosTaurus": "Bos taurus",
    "DelphinapterusLeucas": "Delphinapterus leucas",
    "EquusCaballus": "Equus caballus",
    "HeterocephalusGlaber": "Heterocephalus glaber",
    "HomoSapiens": "Homo sapiens",
    "MacacaMulatta": "Macaca mulatta",
    "MusMusculus": "Mus musculus",
    "OvisAries": "Ovis aries",
    "PapioHamadryas": "Papio hamadryas",
    "RattusNorvegicus": "Rattus norvegicus",
}
POOLED_COMPONENT = "alltissueconcat"

DIRECT_INPUT_ROOT = Path("/path/to/your/file")  # fill in your path
COMPONENT_TRAINING_ROOT = Path("/path/to/your/file")  # fill in your path
MLP_LOO_ROOT = Path("results/figure3/species/mlp_elu_loo_predictions")
ENCV_LOO_INPUT_ROOT = Path("/path/to/your/file")  # fill in your path
UNIVERSAL_CLOCK_MAE = Path("/path/to/your/file")  # fill in your path
PAIRWISE_DISTANCE = Path("/path/to/your/file")  # fill in your path
LIFE_HISTORY = Path("/path/to/your/file")  # fill in your path


def direct_path(species: str) -> Path:
    """Return the direct CpG matrix for one test species."""
    return DIRECT_INPUT_ROOT / f"{species}_fewshot_40k.parquet"


def component_training_path(component: str) -> Path:
    """Return the CpG/age table used to train one MLP-ELU component."""
    return COMPONENT_TRAINING_ROOT / f"{component}_fewshot_40k.parquet"


def mlp_loo_path(species: str) -> Path:
    """Return ten MLP-ELU predictions aligned to a test-species matrix."""
    return MLP_LOO_ROOT / f"loo_per_species_predictions_{species}.parquet"


def encv_loo_path(species: str) -> Path:
    """Return ENCV component predictions used only by the evolution panels."""
    return ENCV_LOO_INPUT_ROOT / f"loo_per_species_predictions_{species}.parquet"
