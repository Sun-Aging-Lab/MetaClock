"""Plot component ENCV coefficient against molecular evolutionary distance."""

import argparse
from pathlib import Path

from generalization.species.evolution_plot_common import (
    DEFAULT_COEFFICIENTS,
    plot_evolution_relationship,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(
            "figures/figure3/generalization/species/component_en_coef_vs_molecular_evolutionary_distance"
        ),
    )
    args = parser.parse_args()
    outputs = plot_evolution_relationship(
        "distance",
        "Molecular evolutionary distance",
        args.output_stem,
        coefficient_path=args.coefficients,
    )
    print(*outputs, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
