"""Train one Figure 3 array-platform dataset/training-size point."""

from generalization.array.datasets import DATASETS, DEFAULT_SHARDS
from generalization.group_cli import training_main


def main() -> int:
    return training_main(DATASETS, DEFAULT_SHARDS, __doc__ or "")


if __name__ == "__main__":
    raise SystemExit(main())
