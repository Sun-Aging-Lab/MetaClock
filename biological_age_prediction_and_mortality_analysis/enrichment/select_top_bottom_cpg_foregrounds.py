#!/usr/bin/env python3
"""Select the top and bottom 1,000 weighted CpGs within the array union."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ENRICHMENT
from ..provenance import file_record, sha256, write_json

DEFAULT_CHRONO_RANKING = Path("results/enrichment/rankings/chrono_cpg_weights.csv.gz")
DEFAULT_BIO_RANKING = Path("results/enrichment/rankings/bio_cpg_weights.csv.gz")
DEFAULT_CHRONO_AUDIT = Path("results/enrichment/rankings/chrono_ranking_summary.json")
DEFAULT_BIO_AUDIT = Path("results/enrichment/rankings/bio_ranking_summary.json")
DEFAULT_MEMBERSHIP = Path("results/enrichment/array_union/platform_membership.csv.gz")
DEFAULT_OUTPUT = Path("results/enrichment/inputs")


TOP_N = int(ENRICHMENT["top_n"])


def parse_args() -> argparse.Namespace:
    """Parse ranking, universe, provenance, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrono-ranking", type=Path, default=DEFAULT_CHRONO_RANKING)
    parser.add_argument("--bio-ranking", type=Path, default=DEFAULT_BIO_RANKING)
    parser.add_argument("--chrono-audit", type=Path, default=DEFAULT_CHRONO_AUDIT)
    parser.add_argument("--bio-audit", type=Path, default=DEFAULT_BIO_AUDIT)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_ranking_provenance(path: Path, ranking_path: Path, model: str) -> None:
    """Bind a component manifest and SHA-256 audit to one ranking file."""
    import json

    audit = json.loads(path.read_text(encoding="utf-8"))
    expected_key = (
        "chronological_affine_components" if model == "chrono" else "biological_affine_components"
    )
    if audit.get("included_components") != ENRICHMENT[expected_key]:
        raise ValueError(f"{model} ranking components do not match the analysis manifest.")
    if audit.get("outputs", {}).get("ranking", {}).get("sha256") != sha256(ranking_path):
        raise ValueError(f"{model} ranking SHA-256 does not match its provenance audit.")


def write_bed(table: pd.DataFrame, path: Path) -> None:
    """Write one-base CpGs using zero-based, half-open BED coordinates."""
    table.assign(start=table["position"].astype(np.int64) - 1, end=table["position"])[
        ["chromosome", "start", "end", "cpg"]
    ].to_csv(path, sep="\t", header=False, index=False)


def select_foregrounds(
    ranking_path: Path,
    union: pd.DataFrame,
    output_dir: Path,
    top_n: int = TOP_N,
) -> dict[str, object]:
    """Select deterministic extremes after filtering a full ranking to the union."""
    ranking = pd.read_csv(ranking_path, usecols=["cpg", "weight"])
    ranking["cpg"] = ranking["cpg"].astype(str).str.strip()
    ranking["weight"] = pd.to_numeric(ranking["weight"], errors="raise")
    if ranking["cpg"].duplicated().any() or not np.isfinite(ranking["weight"]).all():
        raise ValueError(f"Invalid ranking: {ranking_path}")
    lookup = union.set_index("cpg", verify_integrity=True)
    eligible = ranking[ranking["cpg"].isin(lookup.index)].copy()
    selections = {
        "top": eligible.sort_values(["weight", "cpg"], ascending=[False, True]).head(top_n),
        "bottom": eligible.sort_values(["weight", "cpg"], ascending=[True, True]).head(top_n),
    }
    if any(len(selected) != top_n for selected in selections.values()):
        raise ValueError(f"Ranking has fewer than {2 * top_n:,} union-eligible CpGs.")
    if set(selections["top"]["cpg"]).intersection(selections["bottom"]["cpg"]):
        raise ValueError("Top and bottom foregrounds overlap.")
    audit: dict[str, object] = {"ranking_rows": len(ranking), "union_eligible_rows": len(eligible)}
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
    """Validate provenance and create all four GREAT foregrounds."""
    args = parse_args()
    validate_ranking_provenance(args.chrono_audit, args.chrono_ranking, "chrono")
    validate_ranking_provenance(args.bio_audit, args.bio_ranking, "bio")
    union = pd.read_csv(args.membership)
    if len(union) != 896_980 or union["cpg"].duplicated().any():
        raise ValueError("Expected the 896,980-CpG 27K/450K/EPIC union.")
    audits: dict[str, object] = {
        "selection_universe": "27K v1.2 + 450K v1.2 + EPIC v1.0 B4 hg19 union",
        "union_cpgs": len(union),
        "foreground_450k_restriction": False,
        "top_n_per_direction": TOP_N,
        "models": {},
    }
    paths = {"chrono": args.chrono_ranking, "bio": args.bio_ranking}
    for model in ENRICHMENT["models"]:
        audits["models"][model] = select_foregrounds(paths[model], union, args.output_dir / model)
        audits["models"][model]["ranking"] = file_record(paths[model])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "foreground_selection_audit.json", audits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
