"""Strictly aggregate the 30 x 50 healthy-tissue result grid."""

from generalization.group_cli import aggregation_main
from generalization.tissue.datasets import DATASETS, DEFAULT_SHARDS, DEFAULT_SUMMARIES


def main() -> int:
    return aggregation_main(DATASETS, DEFAULT_SHARDS, DEFAULT_SUMMARIES, __doc__ or "")


if __name__ == "__main__":
    raise SystemExit(main())
