"""Strictly aggregate the 30 x 50 array-platform result grid."""

from generalization.array.datasets import DATASETS, DEFAULT_SHARDS, DEFAULT_SUMMARIES
from generalization.group_cli import aggregation_main


def main() -> int:
    return aggregation_main(DATASETS, DEFAULT_SHARDS, DEFAULT_SUMMARIES, __doc__ or "")


if __name__ == "__main__":
    raise SystemExit(main())
