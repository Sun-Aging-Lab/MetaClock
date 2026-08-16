"""Plot final Figure 3 array-platform panels as PDF and SVG."""

from generalization.array.datasets import DATASETS, DEFAULT_FIGURES, DEFAULT_SUMMARIES
from generalization.plotting import plot_group_main


def main() -> int:
    return plot_group_main(DATASETS, DEFAULT_SUMMARIES, DEFAULT_FIGURES, __doc__ or "")


if __name__ == "__main__":
    raise SystemExit(main())
