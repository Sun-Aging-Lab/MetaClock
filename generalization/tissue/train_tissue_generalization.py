"""Train one Figure 3 healthy-tissue dataset/training-size point."""

from generalization.group_cli import training_main
from generalization.tissue.datasets import DATASETS, DEFAULT_SHARDS


def main() -> int:
    return training_main(DATASETS, DEFAULT_SHARDS, __doc__ or "")


if __name__ == "__main__":
    raise SystemExit(main())
