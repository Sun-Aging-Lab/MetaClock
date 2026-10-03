"""Configure Matplotlib to keep its runtime cache inside the repository."""

from __future__ import annotations

import os

from intervention_analysis.paths import PROJECT_ROOT


MPLCONFIGDIR = PROJECT_ROOT / "data" / "intervention" / "matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
