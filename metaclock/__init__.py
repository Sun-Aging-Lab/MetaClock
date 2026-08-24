"""MetaClock: fit meta-regressors from pyaging component clocks."""

from ._version import __version__
from .api import fit
from .components import PyAgingTransformer, predict_components
from .models import available_architectures, make_estimator
from .presets import CHRONOLOGICAL_16_COMPONENT_MODELS

__all__ = [
    "PyAgingTransformer",
    "CHRONOLOGICAL_16_COMPONENT_MODELS",
    "__version__",
    "available_architectures",
    "fit",
    "make_estimator",
    "predict_components",
]
