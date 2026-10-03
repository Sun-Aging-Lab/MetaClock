"""Configure the shared repository-local Matplotlib cache."""

from __future__ import annotations

import os

from repository_paths import PROJECT_ROOT


MPLCONFIGDIR = PROJECT_ROOT / "data" / "matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
