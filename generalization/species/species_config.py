"""Shared species identities and paths for the Figure 3 workflow."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GSE223748_DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "GSE223748"
GSE223748_BETA_PATH = GSE223748_DOWNLOAD_DIR / "GSE223748_datBetaNormalized.csv.gz"
GSE223748_SERIES_PATH = GSE223748_DOWNLOAD_DIR / "GSE223748_series_matrix.txt.gz"
GESTATION_TABLE = Path(__file__).with_name("gse223748_gestation_years.csv")
RESULT_ROOT = PROJECT_ROOT / "results" / "generalization" / "species"
SUBMODEL_ROOT = RESULT_ROOT / "submodels"
INPUT_ROOT = SUBMODEL_ROOT / "inputs"
MODEL_ROOT = SUBMODEL_ROOT / "models"
LOO_ROOT = SUBMODEL_ROOT / "loo_features"

SPECIES = (
    "BosTaurus",
    "CanisLupusFamiliaris",
    "EquusCaballus",
    "HeterocephalusGlaber",
    "HomoSapiens",
    "MacacaMulatta",
    "MusMusculus",
    "OvisAries",
    "PapioHamadryas",
    "RattusNorvegicus",
)
COMPONENT_ORDER = (
    "MusMusculus",
    "HomoSapiens",
    "CanisLupusFamiliaris",
    "RattusNorvegicus",
    "BosTaurus",
    "OvisAries",
    "HeterocephalusGlaber",
    "EquusCaballus",
    "PapioHamadryas",
    "MacacaMulatta",
)
DISPLAY_NAMES = {
    "BosTaurus": "Bos taurus",
    "CanisLupusFamiliaris": "Canis lupus familiaris",
    "EquusCaballus": "Equus caballus",
    "HeterocephalusGlaber": "Heterocephalus glaber",
    "HomoSapiens": "Homo sapiens",
    "MacacaMulatta": "Macaca mulatta",
    "MusMusculus": "Mus musculus",
    "OvisAries": "Ovis aries",
    "PapioHamadryas": "Papio hamadryas",
    "RattusNorvegicus": "Rattus norvegicus",
}
LATIN_TO_MODEL = {value: key for key, value in DISPLAY_NAMES.items()}
EXPECTED_SAMPLE_COUNTS = {
    "BosTaurus": 455,
    "CanisLupusFamiliaris": 742,
    "EquusCaballus": 350,
    "HeterocephalusGlaber": 411,
    "HomoSapiens": 1126,
    "MacacaMulatta": 274,
    "MusMusculus": 3832,
    "OvisAries": 430,
    "PapioHamadryas": 325,
    "RattusNorvegicus": 601,
}
POOLED_COMPONENT = "alltissueconcat"
INPUT_CONFIG_ID = "species-dogtop10-gestation-mixed-precision-v1"
LOO_CONFIG_ID = "species-dogtop10-ordered-loo-v1"
GENERALIZATION_CONFIG_ID = "species-dogtop10-fold-local-encv30-svr-ols-v2"



def direct_input_path(species: str) -> Path:
    return INPUT_ROOT / "direct" / f"{species}_fewshot_40k.parquet"


def metadata_path(species: str) -> Path:
    return INPUT_ROOT / "metadata" / f"{species}.tsv"


def model_dir(model_type: str, source: str) -> Path:
    return MODEL_ROOT / model_type / source


def loo_feature_path(species: str) -> Path:
    return LOO_ROOT / f"loo_features_{species}.parquet"
