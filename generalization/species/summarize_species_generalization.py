"""Strictly aggregate all species 30 x 50 result grids."""

from __future__ import annotations

import argparse
from pathlib import Path

from generalization.aggregation import aggregate_dataset
from generalization.species.species_config import SPECIES
from generalization.species.train_species_generalization import (
    DEFAULT_OUTPUT,
    species_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=SPECIES, action="append")
    parser.add_argument("--shard-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("results/figure3/generalization/species/summaries"),
    )
    args = parser.parse_args()
    for species in args.species or list(SPECIES):
        print(aggregate_dataset(species_spec(species), args.shard_root, args.summary_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
