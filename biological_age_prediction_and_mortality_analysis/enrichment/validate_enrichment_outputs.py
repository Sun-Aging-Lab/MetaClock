#!/usr/bin/env python3
"""Validate array liftOver, ranking selection, GREAT audits, and enrichment vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ENRICHMENT, EXCLUDED_MODELS
from ..provenance import sha256

DEFAULT_MEMBERSHIP = Path("results/enrichment/array_union/platform_membership.csv.gz")
DEFAULT_INPUTS = Path("results/enrichment/inputs")
DEFAULT_RANKINGS = Path("results/enrichment/rankings")
DEFAULT_RESULTS = Path("results/enrichment/rgreat")
DEFAULT_FIGURES = Path("results/enrichment/figures")
DEFAULT_PUBLISHED_WEIGHTS = Path(__file__).with_name("top_bottom_1000_cpg_weights.csv")
POINTS_PER_MM = 72 / 25.4


def read_parameters(path: Path) -> dict[str, str]:
    """Parse a unique-key two-column tab-separated parameter file."""
    rows = [
        line.rstrip("\n").split("\t", 1) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if any(len(row) != 2 for row in rows) or len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"Malformed or duplicate parameter key in {path}.")
    return dict(rows)


def validate_union(membership_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate union identity, coordinates, memberships, and the liftOver audit."""
    membership = pd.read_csv(membership_path)
    required = {"cpg", "chromosome", "position", *ENRICHMENT["platform_counts"]}
    if missing := required.difference(membership.columns):
        raise KeyError(f"Membership is missing columns: {sorted(missing)}")
    if len(membership) != int(ENRICHMENT["union_cpgs"]):
        raise ValueError("Array union has the wrong number of CpGs.")
    if (
        membership["cpg"].duplicated().any()
        or membership.duplicated(["chromosome", "position"]).any()
    ):
        raise ValueError("Array union has a duplicate CpG ID or genomic coordinate.")
    observed: dict[str, int] = {}
    for column, expected in ENRICHMENT["platform_counts"].items():
        if not membership[column].isin([True, False, 0, 1]).all():
            raise ValueError(f"{column} is not a Boolean membership field.")
        observed[column] = int(membership[column].astype(bool).sum())
        if observed[column] != int(expected):
            raise ValueError(f"Unexpected {column} count: {observed[column]}.")
    audit_path = membership_path.parent / "annotation_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("outputs", {}).get("membership", {}).get("sha256") != sha256(membership_path):
        raise ValueError("Array membership does not match annotation_audit.json.")
    if audit.get("annotation_packages") != {
        "IlluminaHumanMethylation27kanno.ilmn12.hg19": "0.6.0",
        "IlluminaHumanMethylation450kanno.ilmn12.hg19": "0.6.1",
        "IlluminaHumanMethylationEPICanno.ilm10b4.hg19": "0.6.0",
    }:
        raise ValueError("Array annotation packages are not the pinned 27K/450K/EPIC manifests.")
    liftover, cross = audit.get("liftover_27k", {}), audit.get("cross_platform", {})
    if liftover.get("status") != "applied_hg18_to_hg19":
        raise ValueError("27K v1.2 coordinates were not lifted from hg18 to hg19.")
    if liftover.get("matches_450k_before", 10**9) > 100:
        raise ValueError("27K pre-lift coordinate audit lacks the expected hg18 signature.")
    if liftover.get("matches_450k_after") != liftover.get("shared_450k_after"):
        raise ValueError("Lifted 27K coordinates do not fully agree with 450K v1.2.")
    if liftover.get("uniquely_lifted_regions") != 27574 or liftover.get("unmapped_regions") != 4:
        raise ValueError("27K liftOver counts differ from the formal audit.")
    if (
        cross.get("identifier_coordinate_conflicts") != 0
        or cross.get("duplicate_union_coordinates") != 0
    ):
        raise ValueError("Array union retains an identifier or coordinate conflict.")
    return membership, observed


def read_bed(path: Path) -> pd.DataFrame:
    """Read and validate a four-column one-base BED table."""
    table = pd.read_csv(path, sep="\t", header=None, names=["chromosome", "start", "end", "cpg"])
    if (
        table.isna().any().any()
        or table["cpg"].duplicated().any()
        or not table["end"].eq(table["start"] + 1).all()
    ):
        raise ValueError(f"Invalid one-base foreground BED: {path}")
    return table


def validate_foregrounds(membership: pd.DataFrame, inputs_dir: Path, rankings_dir: Path) -> int:
    """Recompute deterministic top/bottom selections and verify BED coordinates."""
    audit = json.loads((inputs_dir / "foreground_selection_audit.json").read_text(encoding="utf-8"))
    if (
        audit.get("foreground_450k_restriction") is not False
        or audit.get("top_n_per_direction") != ENRICHMENT["top_n"]
    ):
        raise ValueError("Foreground audit has a 450K restriction or wrong selection size.")
    if audit.get("excluded_component") not in EXCLUDED_MODELS:
        raise ValueError("Foreground audit does not enforce the global dnamfitage exclusion.")
    lookup = membership.set_index("cpg", verify_integrity=True)
    total = 0
    for model in ENRICHMENT["models"]:
        ranking_path = rankings_dir / f"{model}_cpg_weights.csv.gz"
        ranking_audit = json.loads(
            (rankings_dir / f"{model}_ranking_summary.json").read_text(encoding="utf-8")
        )
        component_key = (
            "chronological_affine_components"
            if model == "chrono"
            else "biological_affine_components"
        )
        if ranking_audit.get("included_components") != ENRICHMENT[component_key]:
            raise ValueError(f"{model} ranking uses an unexpected component manifest.")
        if set(ranking_audit.get("globally_excluded_models", [])) != set(EXCLUDED_MODELS):
            raise ValueError(f"{model} ranking lacks the global exclusion.")
        if ranking_audit.get("outputs", {}).get("ranking", {}).get("sha256") != sha256(
            ranking_path
        ):
            raise ValueError(f"{model} ranking hash does not match its provenance.")
        ranking = pd.read_csv(ranking_path)
        ranking = ranking[ranking["cpg"].isin(lookup.index)].copy()
        ranking["weight"] = pd.to_numeric(ranking["weight"], errors="raise")
        if ranking["cpg"].duplicated().any() or not np.isfinite(ranking["weight"]).all():
            raise ValueError(f"{model} ranking is duplicated or non-finite.")
        expected = {
            "top": ranking.sort_values(["weight", "cpg"], ascending=[False, True]).head(
                ENRICHMENT["top_n"]
            ),
            "bottom": ranking.sort_values(["weight", "cpg"], ascending=[True, True]).head(
                ENRICHMENT["top_n"]
            ),
        }
        if set(expected["top"]["cpg"]).intersection(expected["bottom"]["cpg"]):
            raise ValueError(f"{model} top and bottom foregrounds overlap.")
        for direction, wanted in expected.items():
            selected = pd.read_csv(inputs_dir / model / f"{direction}_selected_cpgs.csv")
            if selected["cpg"].tolist() != wanted["cpg"].tolist():
                raise ValueError(f"{model}/{direction} is not the deterministic ranking extreme.")
            bed = read_bed(inputs_dir / model / f"{direction}_foreground.bed")
            expected_bed = lookup.loc[bed["cpg"]]
            if (
                not bed["chromosome"]
                .astype(str)
                .eq(expected_bed["chromosome"].astype(str).to_numpy())
                .all()
            ):
                raise ValueError(f"{model}/{direction} chromosome mismatch against union.")
            if not bed["start"].eq(expected_bed["position"].to_numpy() - 1).all():
                raise ValueError(f"{model}/{direction} coordinate mismatch against union.")
            total += len(selected)
    return total


def expected_parameters() -> dict[str, str]:
    """Return the manifest-derived rGREAT parameter contract."""
    return {
        "rGREAT_version": str(ENRICHMENT["rGREAT_version"]),
        "genome": str(ENRICHMENT["genome"]),
        "min_gene_set_size": str(ENRICHMENT["min_gene_set_size"]),
        "min_region_hits": str(ENRICHMENT["min_region_hits"]),
        "mode": str(ENRICHMENT["mode"]),
        "extend_from": str(ENRICHMENT["extend_from"]),
        "basal_upstream_bp": str(ENRICHMENT["basal_upstream_bp"]),
        "basal_downstream_bp": str(ENRICHMENT["basal_downstream_bp"]),
        "extension_bp": str(ENRICHMENT["extension_bp"]),
        "exclude": str(ENRICHMENT["exclude"]),
        "background_regions": str(ENRICHMENT["union_cpgs"]),
        "foreground_regions": str(ENRICHMENT["top_n"]),
    }


def validate_mapping_audit(directory: Path, library: str) -> None:
    """Require term-level unique Entrez mapping audits for non-GO libraries."""
    mapping = pd.read_csv(directory / "gene_set_symbol_entrez_mapping.csv")
    term_audit_path = directory / "gene_set_term_mapping_audit.csv"
    if library == "go_bp":
        if not mapping.empty or term_audit_path.exists():
            raise ValueError("GO:BP must use native GO identifiers, not SYMBOL-to-Entrez mapping.")
        return
    term_audit = pd.read_csv(term_audit_path)
    mapped = mapping[mapping["mapped"].astype(str).str.lower().isin(["true", "1"])]
    if mapped.duplicated(["term", "entrez_id"]).any():
        raise ValueError(f"{library} mapping contains duplicate Entrez IDs within a term.")
    counts = mapped.groupby("term")["entrez_id"].nunique()
    audited = term_audit.set_index("term")["mapped_unique_entrez"].astype(int)
    if not audited.equals(counts.reindex(audited.index, fill_value=0).astype(int)):
        raise ValueError(f"{library} term mapping audit does not match deduplicated mappings.")


def validate_results(results_dir: Path) -> int:
    """Validate every enrichment table, filtered subset, association, and parameter file."""
    total = 0
    required = {"p_value", "p_adjust", "observed_region_hits"}
    wanted_parameters = expected_parameters()
    for model in ENRICHMENT["models"]:
        for library in ENRICHMENT["libraries"]:
            validate_mapping_audit(results_dir / model / library, library)
            for direction in ENRICHMENT["directions"]:
                directory = results_dir / model / library / direction
                parameters = read_parameters(directory / "parameters.tsv")
                mismatch = {
                    key: (parameters.get(key), value)
                    for key, value in wanted_parameters.items()
                    if parameters.get(key) != value
                }
                if mismatch:
                    raise ValueError(f"Unexpected rGREAT parameters in {directory}: {mismatch}")
                all_terms = pd.read_csv(directory / f"{library}_all.csv")
                filtered = pd.read_csv(directory / f"{library}_fdr05.csv")
                if all_terms.empty or not required.issubset(all_terms.columns):
                    raise ValueError(f"Invalid enrichment table: {directory}")
                values = all_terms[list(required)].apply(pd.to_numeric, errors="raise")
                if (
                    not np.isfinite(values).all().all()
                    or not values[["p_value", "p_adjust"]]
                    .apply(lambda c: c.between(0, 1))
                    .all()
                    .all()
                ):
                    raise ValueError(
                        f"Non-finite or out-of-range enrichment statistics: {directory}"
                    )
                if not values["observed_region_hits"].ge(ENRICHMENT["min_region_hits"]).all():
                    raise ValueError(f"{directory} violates min_region_hits.")
                expected_filtered = all_terms[all_terms["p_adjust"].lt(0.05)].reset_index(drop=True)
                if not filtered.reset_index(drop=True).equals(expected_filtered):
                    raise ValueError(f"FDR table is not the exact BH<0.05 subset: {directory}")
                associations = pd.read_csv(directory / "region_gene_associations.csv")
                if (
                    associations.empty
                    or "cpg" not in associations
                    or associations["cpg"].isna().any()
                ):
                    raise ValueError(f"Missing region-gene association audit: {directory}")
                total += 1
    return total


def validate_vectors(figure_dir: Path) -> int:
    """Require the eight semantic 150 x 75 mm PDF/SVG vector pairs."""
    import pymupdf

    count = 0
    library_names = {
        "go_bp": "go_biological_process",
        **{key: key for key in ENRICHMENT["libraries"] if key != "go_bp"},
    }
    for model in ENRICHMENT["models"]:
        model_name = "chronological" if model == "chrono" else "biological"
        for library in ENRICHMENT["libraries"]:
            stem = f"enrichment_{model_name}_{library_names[library]}_top_bottom_1000"
            pdf, svg = figure_dir / f"{stem}.pdf", figure_dir / f"{stem}.svg"
            if not pdf.is_file() or not svg.is_file():
                raise FileNotFoundError(f"Missing vector pair: {stem}")
            document = pymupdf.open(pdf)
            expected = (150 * POINTS_PER_MM, 75 * POINTS_PER_MM)
            if len(document) != 1 or any(
                abs(a - b) > 0.02
                for a, b in zip(
                    (document[0].rect.width, document[0].rect.height), expected, strict=True
                )
            ):
                raise ValueError(f"Invalid PDF canvas: {pdf}")
            text = document[0].get_text() + svg.read_text(encoding="utf-8")
            if "dnamfitage" in text.lower() or "Liberation Sans" not in text:
                raise ValueError(f"Excluded component or wrong font in {stem}.")
            count += 1
    return count


def validate_published_weights(path: Path) -> int:
    """Validate the compact top/bottom CpG-weight publication table."""
    table = pd.read_csv(path)
    required = ["group", "direction", "rank", "cpg", "weight"]
    if table.columns.tolist() != required:
        raise ValueError("Published CpG weights do not have the required ordered schema.")
    groups = {"chronological", "biological"}
    if set(table["group"]) != groups or set(table["direction"]) != set(
        ENRICHMENT["directions"]
    ):
        raise ValueError("Published CpG weights contain an unexpected group or direction.")
    for keys, subset in table.groupby(["group", "direction"], sort=False):
        if len(subset) != ENRICHMENT["top_n"] or subset["cpg"].duplicated().any():
            raise ValueError(f"Published CpG weight group {keys} is incomplete or duplicated.")
        if subset["rank"].tolist() != list(range(1, ENRICHMENT["top_n"] + 1)):
            raise ValueError(f"Published CpG weight ranks are invalid for {keys}.")
        weights = pd.to_numeric(subset["weight"], errors="raise").to_numpy(float)
        if not np.isfinite(weights).all():
            raise ValueError(f"Published CpG weights are non-finite for {keys}.")
        expected = np.sort(weights)[::-1] if keys[1] == "top" else np.sort(weights)
        if not np.array_equal(weights, expected):
            raise ValueError(f"Published CpG weights are not sorted for {keys}.")
    return len(table)


def main() -> int:
    """Run focused validators and write a concise machine-readable summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--rankings-dir", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--published-weights", type=Path, default=DEFAULT_PUBLISHED_WEIGHTS)
    args = parser.parse_args()
    membership, counts = validate_union(args.membership)
    report: dict[str, Any] = {
        "union_cpgs": len(membership),
        "platform_counts": counts,
        "foreground_rows": validate_foregrounds(membership, args.inputs_dir, args.rankings_dir),
        "result_tables": validate_results(args.results_dir),
        "vector_pairs": validate_vectors(args.figure_dir),
        "published_weight_rows": validate_published_weights(args.published_weights),
        "globally_excluded_component": "dnamfitage",
    }
    (args.figure_dir / "validation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
