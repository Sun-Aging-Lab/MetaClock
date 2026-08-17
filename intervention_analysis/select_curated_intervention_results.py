#!/usr/bin/env python3
"""Apply the audited one-representative-per-intervention selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results/intervention/effects/all_intervention_effects.csv"
DEFAULT_SELECTION_AUDIT = Path("/path/to/your/file")  # fill in your path
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/intervention/selected"
DEFAULT_EXCLUDED_GSE_IDS = (
    "GSE276988",
    "GSE330869",
    "GSE168628",
    "GSE130028",
)


def selected_mask(values: pd.Series) -> pd.Series:
    """Parse explicit Boolean audit values and reject ambiguous spreadsheet entries."""
    if values.isna().any():
        raise ValueError("Selection audit contains missing selected values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    invalid = normalized.loc[~normalized.isin({"true", "false"})]
    if not invalid.empty:
        raise ValueError(
            "selected must contain only true/false Boolean values; invalid rows: "
            f"{invalid.index.tolist()}"
        )
    return normalized.eq("true")


def select_results(
    results: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    excluded_gse_ids: tuple[str, ...] = DEFAULT_EXCLUDED_GSE_IDS,
) -> pd.DataFrame:
    """Select representatives, exclude the formal dataset set, and recompute BH."""
    required_results = {"comparison_id", "two_sided_p_value"}
    required_audit = {"comparison_id", "intervention_label", "selected"}
    if missing := required_results.difference(results.columns):
        raise KeyError(f"Results are missing columns: {sorted(missing)}")
    if missing := required_audit.difference(audit.columns):
        raise KeyError(f"Selection audit is missing columns: {sorted(missing)}")
    if results["comparison_id"].duplicated().any():
        raise ValueError("Completed results contain duplicate comparison_id values")
    chosen = audit.loc[
        selected_mask(audit["selected"]), ["comparison_id", "intervention_label"]
    ].copy()
    if chosen["comparison_id"].duplicated().any():
        raise ValueError("Selection audit chooses a comparison more than once.")
    selected = chosen.merge(results, on="comparison_id", how="left", validate="one_to_one")
    if selected["two_sided_p_value"].isna().any():
        raise ValueError("One or more selected comparisons lack completed results.")
    if excluded_gse_ids:
        if "gse_id" not in selected:
            raise KeyError("Results must contain gse_id for dataset exclusion")
        selected = selected.loc[~selected["gse_id"].astype(str).isin(excluded_gse_ids)].copy()
    if selected.empty:
        raise ValueError("No selected comparisons remain after dataset exclusion")
    p_values = pd.to_numeric(selected["two_sided_p_value"], errors="raise")
    if selected.empty or not p_values.between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Selected comparisons require finite two-sided p-values in [0, 1]")
    selected["bh_q_value_after_dataset_exclusion"] = multipletests(p_values, method="fdr_bh")[1]
    return selected.sort_values(
        ["bh_q_value_after_dataset_exclusion", "two_sided_p_value"]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--selection-audit", type=Path, default=DEFAULT_SELECTION_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude-gse-id",
        action="append",
        default=None,
        help="GSE ID excluded before the final BH correction (repeatable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = pd.read_csv(args.selection_audit)
    excluded_gse_ids = tuple(
        DEFAULT_EXCLUDED_GSE_IDS if args.exclude_gse_id is None else args.exclude_gse_id
    )
    selected = select_results(
        pd.read_csv(args.results),
        audit,
        excluded_gse_ids=excluded_gse_ids,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "selected_intervention_results.csv", index=False)
    audit.to_csv(args.output_dir / "intervention_selection_audit.csv", index=False)
    summary = {
        "selected_interventions": int(len(selected)),
        "fdr_le_0_05": int(selected["bh_q_value_after_dataset_exclusion"].le(0.05).sum()),
        "fdr_le_0_20": int(selected["bh_q_value_after_dataset_exclusion"].le(0.20).sum()),
        "excluded_gse_ids": list(excluded_gse_ids),
        "multiple_testing": (
            "Benjamini-Hochberg across selected interventions after dataset exclusion"
        ),
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
