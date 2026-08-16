"""Precompute the shared 90-row directed ENCV coefficient table once."""

import argparse
from pathlib import Path

from generalization.species.evolution_plot_common import (
    DEFAULT_COEFFICIENTS,
    prepare_coefficient_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_COEFFICIENTS)
    args = parser.parse_args()
    print(prepare_coefficient_table(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
