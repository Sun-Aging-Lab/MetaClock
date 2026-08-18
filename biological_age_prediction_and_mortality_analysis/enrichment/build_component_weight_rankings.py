#!/usr/bin/env python3
"""Aggregate raw-scale affine component CpG coefficients into MetaClock rankings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ENRICHMENT
from ..provenance import file_record, software_versions, write_json

DEFAULT_META_WEIGHTS = Path("/path/to/your/file")  # fill in your path
DEFAULT_COMPONENT_CPG = Path("/path/to/your/file")  # fill in your path
DEFAULT_COMPONENT_AUDIT = Path("/path/to/your/file")  # fill in your path
DEFAULT_OUTPUT = Path("results/enrichment/rankings")


def expected_components(model: str) -> tuple[str, ...]:
    """Return the affine components allowed to contribute CpG weights."""
    key = "chronological_affine_components" if model == "chrono" else "biological_affine_components"
    return tuple(ENRICHMENT[key])


def build_ranking(
    model: str,
    meta_weights: pd.DataFrame,
    component_cpg: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multiply raw-scale meta/component coefficients and sum them per CpG."""
    expected = expected_components(model)
    for name, table, required in (
        ("meta weights", meta_weights, {"model", "clock", "coefficient"}),
        ("component CpGs", component_cpg, {"clock", "cpg", "coefficient"}),
        ("component audit", audit, {"clock", "fixed_affine", "artifact_sha256"}),
    ):
        if missing := required.difference(table.columns):
            raise KeyError(f"{name} are missing columns: {sorted(missing)}")
    weights = meta_weights[meta_weights["model"].eq(model)].copy()
    clocks = weights["clock"].astype(str).tolist()
    if clocks != list(expected) or weights["clock"].duplicated().any():
        raise ValueError(f"{model} meta weights do not match the manifest component order.")
    audit_rows = audit[audit["clock"].isin(expected)].set_index("clock")
    if set(audit_rows.index) != set(expected) or not audit_rows["fixed_affine"].astype(bool).all():
        raise ValueError("Every contributing component must have a unique fixed-affine audit row.")
    if audit_rows["artifact_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").eq(False).any():
        raise ValueError("Every component artifact must have a valid SHA-256 digest.")
    coefficients = component_cpg[component_cpg["clock"].isin(expected)].copy()
    if coefficients.duplicated(["clock", "cpg"]).any():
        raise ValueError("Component CpG coefficient table contains duplicate clock/CpG rows.")
    if set(coefficients["clock"].astype(str)) != set(expected):
        raise ValueError("At least one manifest component has no CpG coefficients.")
    coefficients["coefficient"] = pd.to_numeric(coefficients["coefficient"], errors="raise")
    weights["coefficient"] = pd.to_numeric(weights["coefficient"], errors="raise")
    if (
        not np.isfinite(coefficients["coefficient"]).all()
        or not np.isfinite(weights["coefficient"]).all()
    ):
        raise ValueError("Coefficient tables contain non-finite values.")
    joined = coefficients.merge(
        weights[["clock", "coefficient"]].rename(columns={"coefficient": "meta_coefficient"}),
        on="clock",
        validate="many_to_one",
    )
    joined["contribution"] = joined["coefficient"] * joined["meta_coefficient"]
    ranking = (
        joined.groupby("cpg", as_index=False)["contribution"]
        .sum()
        .rename(columns={"contribution": "weight"})
        .sort_values(["weight", "cpg"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    if ranking["cpg"].duplicated().any() or not np.isfinite(ranking["weight"]).all():
        raise RuntimeError("Aggregated CpG ranking is invalid.")
    return ranking, joined


def main() -> int:
    """Build a chronological or biological ranking from normalized coefficient tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(ENRICHMENT["models"]))
    parser.add_argument("--meta-weights", type=Path, default=DEFAULT_META_WEIGHTS)
    parser.add_argument("--component-cpg", type=Path, default=DEFAULT_COMPONENT_CPG)
    parser.add_argument("--component-audit", type=Path, default=DEFAULT_COMPONENT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = {
        "meta_weights": args.meta_weights,
        "component_cpg": args.component_cpg,
        "component_audit": args.component_audit,
    }
    ranking, contributions = build_ranking(
        args.model,
        pd.read_csv(args.meta_weights),
        pd.read_csv(args.component_cpg),
        pd.read_csv(args.component_audit),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / f"{args.model}_cpg_weights.csv.gz"
    contribution_path = args.output_dir / f"{args.model}_component_cpg_contributions.csv.gz"
    ranking.to_csv(ranking_path, index=False)
    contributions.to_csv(contribution_path, index=False)
    write_json(
        args.output_dir / f"{args.model}_ranking_summary.json",
        {
            "model": args.model,
            "coefficient_scale": "raw input scale; affine intercepts do not affect CpG ranking",
            "included_components": list(expected_components(args.model)),
            "ranking_rows": len(ranking),
            "inputs": {name: file_record(path) for name, path in inputs.items()},
            "outputs": {
                "ranking": file_record(ranking_path),
                "contributions": file_record(contribution_path),
            },
            "software": software_versions(("numpy", "pandas")),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
