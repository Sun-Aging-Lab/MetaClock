#!/usr/bin/env python3
"""Build GREAT foregrounds from the published top/bottom-1,000 CpG table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from repository_paths import PROJECT_ROOT, repo_path

from ..config import ENRICHMENT
from ..provenance import file_record, write_json

DEFAULT_SELECTIONS = Path(__file__).with_name("top_bottom_1000_cpg_weights.csv")
DEFAULT_MEMBERSHIP = PROJECT_ROOT / "results/enrichment/array_union/platform_membership.csv.gz"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/enrichment/inputs"


TOP_N = int(ENRICHMENT["top_n"])
GROUP_BY_MODEL = {"chrono": "chronological", "bio": "biological"}


def parse_args() -> argparse.Namespace:
    """Parse published selections, union membership, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-cpgs", type=repo_path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--membership", type=repo_path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--output-dir", type=repo_path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_bed(table: pd.DataFrame, path: Path) -> None:
    """Write one-base CpGs using zero-based, half-open BED coordinates."""
    table.assign(start=table["position"].astype(np.int64) - 1, end=table["position"])[
        ["chromosome", "start", "end", "cpg"]
    ].to_csv(path, sep="\t", header=False, index=False)


def select_foregrounds(
    published: pd.DataFrame,
    group: str,
    union: pd.DataFrame,
    output_dir: Path,
    top_n: int = TOP_N,
) -> dict[str, object]:
    """Validate and write one model's published top and bottom foregrounds."""
    model_rows = published[published["group"].eq(group)].copy()
    lookup = union.set_index("cpg", verify_integrity=True)
    selections = {
        direction: model_rows[model_rows["direction"].eq(direction)].sort_values("rank")
        for direction in ("top", "bottom")
    }
    for direction, selected in selections.items():
        if len(selected) != top_n or selected["rank"].tolist() != list(range(1, top_n + 1)):
            raise ValueError(f"{group}/{direction} must contain ranks 1 through {top_n}.")
        if selected["cpg"].duplicated().any():
            raise ValueError(f"{group}/{direction} contains duplicate CpGs.")
        monotonic = (
            selected["weight"].is_monotonic_decreasing
            if direction == "top"
            else selected["weight"].is_monotonic_increasing
        )
        if not monotonic:
            raise ValueError(f"{group}/{direction} weights do not follow rank order.")
        missing = set(selected["cpg"]).difference(lookup.index)
        if missing:
            raise ValueError(f"{group}/{direction} contains {len(missing)} CpGs outside the union.")
    if set(selections["top"]["cpg"]).intersection(selections["bottom"]["cpg"]):
        raise ValueError("Top and bottom foregrounds overlap.")
    audit: dict[str, object] = {"published_rows": len(model_rows)}
    output_dir.mkdir(parents=True, exist_ok=True)
    for direction, selected in selections.items():
        selected = selected.join(lookup, on="cpg", validate="one_to_one")
        selected.to_csv(output_dir / f"{direction}_selected_cpgs.csv", index=False)
        write_bed(
            selected.sort_values(["chromosome", "position", "cpg"]),
            output_dir / f"{direction}_foreground.bed",
        )
        audit[direction] = {
            "regions": len(selected),
            "in_27k": int(selected["in_27k"].sum()),
            "in_450k": int(selected["in_450k"].sum()),
            "in_epic": int(selected["in_epic"].sum()),
        }
    return audit


def main() -> int:
    """Validate the publication table and create all four GREAT foregrounds."""
    args = parse_args()
    published = pd.read_csv(args.selected_cpgs)
    required = {"group", "direction", "rank", "cpg", "weight"}
    if missing := required.difference(published.columns):
        raise KeyError(f"Published CpG table is missing columns: {sorted(missing)}")
    published = published[list(required)].copy()
    published["cpg"] = published["cpg"].astype(str).str.strip()
    published["rank"] = pd.to_numeric(published["rank"], errors="raise")
    published["weight"] = pd.to_numeric(published["weight"], errors="raise")
    if not np.isfinite(published[["rank", "weight"]]).all().all():
        raise ValueError("Published ranks and weights must be finite.")
    if not published["rank"].eq(np.floor(published["rank"])).all():
        raise ValueError("Published ranks must be integers.")
    published["rank"] = published["rank"].astype(int)
    expected_groups = set(GROUP_BY_MODEL.values())
    if set(published["group"]) != expected_groups or set(published["direction"]) != {
        "top",
        "bottom",
    }:
        raise ValueError("Published table must contain chronological/biological top/bottom groups.")
    union = pd.read_csv(args.membership)
    if len(union) != 896_980 or union["cpg"].duplicated().any():
        raise ValueError("Expected the 896,980-CpG 27K/450K/EPIC union.")
    audits: dict[str, object] = {
        "selection_universe": "27K v1.2 + 450K v1.2 + EPIC v1.0 B4 hg19 union",
        "union_cpgs": len(union),
        "foreground_450k_restriction": False,
        "top_n_per_direction": TOP_N,
        "published_selections": file_record(args.selected_cpgs),
        "models": {},
    }
    for model in ENRICHMENT["models"]:
        audits["models"][model] = select_foregrounds(
            published, GROUP_BY_MODEL[model], union, args.output_dir / model
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "foreground_selection_audit.json", audits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
