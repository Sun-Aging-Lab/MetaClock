"""Plot species generalization with Mean and Universal Clock baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generalization.modeling import project_path
from generalization.plotting import plot_generalization_panel
from generalization.species.species_config import SPECIES, UNIVERSAL_CLOCK_MAE
from generalization.species.train_species_generalization import species_spec


def load_universal_mae(path: Path) -> dict[str, float]:
    """Require one finite, non-negative Universal Clock MAE per plotted species."""
    frame = pd.read_csv(project_path(path))
    if set(frame.columns) != {"species", "mae"} or frame["species"].duplicated().any():
        raise ValueError(f"Invalid Universal Clock baseline schema: {path}")
    if set(frame["species"]) != set(SPECIES):
        raise ValueError(f"{path}: species set differs")
    values = pd.to_numeric(frame["mae"], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path}: MAE values are invalid")
    return frame.assign(mae=values).set_index("species")["mae"].to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=SPECIES, action="append")
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("results/figure3/generalization/species/summaries"),
    )
    parser.add_argument("--universal-mae", type=Path, default=UNIVERSAL_CLOCK_MAE)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures/figure3/generalization/species")
    )
    args = parser.parse_args()
    baselines = load_universal_mae(args.universal_mae)
    for species in args.species or list(SPECIES):
        spec = species_spec(species)
        summary = args.summary_root / f"{species}_plot_summary.csv"
        outputs = plot_generalization_panel(
            summary,
            spec,
            args.output_dir / f"{species}_fewshot_clock_with_mean_and_universal_baselines",
            universal_mae=baselines[species],
            italic_title=True,
        )
        print(*outputs, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
