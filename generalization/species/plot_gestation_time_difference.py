"""Plot component ENCV coefficient against absolute gestation-time difference."""

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
            "figures/figure3/generalization/species/component_en_coef_vs_gestation_difference"
        ),
    )
    args = parser.parse_args()
    outputs = plot_evolution_relationship(
        "gestation_difference_days",
        "|Gestation time difference| (days)",
        args.output_stem,
        coefficient_path=args.coefficients,
    )
    print(*outputs, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
