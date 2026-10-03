#!/usr/bin/env python3
"""Build the shared sample-level InCHIANTI analysis table from three raw files."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from repository_paths import relative_path

from .config import (
    DNAM_FEATURE_PREFIX_COUNTS,
    DNAM_FEATURE_PREFIXES,
    INCHIANTI_DNAM_FEATURE_COUNT,
)

MODULE_DIR = Path(__file__).resolve().parent
DNAM_RDATA_PATH = Path("path/to/your/file")  # fill in your path: DNAm RData
LAB_SAS_PATH = Path("path/to/your/file")  # fill in your path: baseline laboratory SAS table
VITAL_SAS_PATH = Path("path/to/your/file")  # fill in your path: Vital Status SAS table
OUTPUT_PATH = MODULE_DIR / "outputs" / "preprocessing" / "inchianti_analysis_ready.parquet"
CRP_SOURCE = "prefer_hs"
APPLY_PLAUSIBLE_RANGE_FILTER = True

FORMULA_SOURCE = {
    "citation": (
        "Levine et al. 2018, Aging (Albany NY), An epigenetic biomarker "
        "of aging for lifespan and healthspan"
    ),
    "url": "https://www.aging-us.com/article/101414/text",
}
COEFFICIENTS = {
    "intercept": -19.90667,
    "albumin_g_L": -0.03359355,
    "creatinine_umol_L": 0.009506491,
    "glucose_mmol_L": 0.1953192,
    "ln_crp_mg_dL": 0.09536762,
    "lymphocyte_pct": -0.01199984,
    "mcv_fL": 0.02676401,
    "rdw_pct": 0.3306156,
    "alkaline_phosphatase_U_L": 0.001868778,
    "wbc_1000_per_uL": 0.05542406,
    "age_years": 0.08035356,
}
PLAUSIBLE_RANGES = {
    "age_years": (20.0, 120.0),
    "albumin_g_L": (10.0, 70.0),
    "creatinine_umol_L": (10.0, 2500.0),
    "glucose_mmol_L": (1.0, 60.0),
    "ln_crp_mg_dL": (-8.0, 8.0),
    "lymphocyte_pct": (0.0, 100.0),
    "mcv_fL": (40.0, 150.0),
    "rdw_pct": (5.0, 40.0),
    "alkaline_phosphatase_U_L": (1.0, 3000.0),
    "wbc_1000_per_uL": (0.1, 200.0),
}
GAMMA_PER_MONTH = 0.0076927
FOLLOWUP_MONTHS = 120.0
AGE_CONVERSION_OFFSET = 141.50225
AGE_CONVERSION_SCALE = 0.090165
AGE_CONVERSION_MULTIPLIER = 0.00553


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    """Require named columns in a source table."""
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def sas_date(values: pd.Series) -> pd.Series:
    """Decode pandas-decoded SAS dates or numeric SAS day counts."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.to_datetime(numeric, unit="D", origin="1960-01-01", errors="coerce")


def load_rdata(path: Path) -> tuple[Any, pd.Index, pd.Index, pd.DataFrame]:
    """Read the canonical beta and sample-metadata objects with rdata."""
    try:
        import rdata
    except ImportError as exc:
        raise RuntimeError("Install the pinned rdata dependency to read InCHIANTI RData.") from exc

    objects = rdata.read_rda(path)
    missing_objects = {"b.baseline", "pheno.baseline"}.difference(objects)
    if missing_objects:
        raise ValueError(f"{path} is missing R objects: {sorted(missing_objects)}")

    beta = objects["b.baseline"]
    if len(beta.dims) != 2:
        raise ValueError(f"b.baseline must be two-dimensional, found {beta.dims}")
    cpg_dim, sample_dim = beta.dims
    cpg_names = pd.Index([str(value) for value in beta.coords[cpg_dim].values], name="CpG")
    sample_ids = pd.Index(
        [str(value) for value in beta.coords[sample_dim].values], name="sampleID"
    )
    if cpg_names.has_duplicates or sample_ids.has_duplicates:
        raise ValueError("b.baseline has duplicate CpG names or sample IDs")

    pheno = objects["pheno.baseline"].copy()
    pheno.columns = [str(column) for column in pheno.columns]
    require_columns(pheno, {"ID", "sampleID"}, path)
    pheno["ID"] = pd.to_numeric(pheno["ID"], errors="coerce")
    pheno = pheno.dropna(subset=["ID", "sampleID"]).copy()
    pheno["sampleID"] = pheno["sampleID"].astype(str)
    if pheno["sampleID"].duplicated().any():
        raise ValueError("pheno.baseline has duplicate sampleID values")
    missing_pheno = sample_ids.difference(pheno["sampleID"])
    if len(missing_pheno):
        raise ValueError(f"{len(missing_pheno)} beta samples have no pheno.baseline row")
    return beta, cpg_names, sample_ids, pheno


def build_clinical_table(
    path: Path, crp_source: str, range_filter: bool
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compute Levine Clinical PhenoAge from the baseline laboratory SAS table."""
    lab = pd.read_sas(path, format="sas7bdat", encoding="latin1")
    lab.columns = [str(column) for column in lab.columns]
    required = {
        "CODE98", "SITE", "SEX", "DATA_NAS", "X_DATEL", "X_AGEL", "X_PT",
        "X_ALB", "X_CREA", "X_GLU", "X_CRP_HS", "X_CRP_LS", "X_P_LIN",
        "X_VGM", "X_IDE", "X_PALK", "X_GB",
    }
    require_columns(lab, required, path)

    out = lab.loc[:, ["CODE98", "SITE", "SEX", "DATA_NAS"]].rename(
        columns={"CODE98": "ID"}
    )
    out["ID"] = pd.to_numeric(out["ID"], errors="coerce")
    out["lab_date"] = sas_date(lab["X_DATEL"])
    out["age_years"] = pd.to_numeric(lab["X_AGEL"], errors="coerce")
    out["albumin_g_L"] = (
        pd.to_numeric(lab["X_PT"], errors="coerce")
        * 10.0
        * pd.to_numeric(lab["X_ALB"], errors="coerce")
        / 100.0
    )
    out["creatinine_umol_L"] = pd.to_numeric(lab["X_CREA"], errors="coerce") * 88.4017
    out["glucose_mmol_L"] = pd.to_numeric(lab["X_GLU"], errors="coerce") / 18.0182
    hs = pd.to_numeric(lab["X_CRP_HS"], errors="coerce")
    ls = pd.to_numeric(lab["X_CRP_LS"], errors="coerce")
    if crp_source == "hs":
        out["crp_ug_mL"] = hs
        out["crp_source"] = np.where(hs.notna(), "X_CRP_HS", pd.NA)
    elif crp_source == "ls":
        out["crp_ug_mL"] = ls
        out["crp_source"] = np.where(ls.notna(), "X_CRP_LS", pd.NA)
    else:
        out["crp_ug_mL"] = hs.combine_first(ls)
        out["crp_source"] = np.select(
            [hs.notna(), ls.notna()], ["X_CRP_HS", "X_CRP_LS"], default=pd.NA
        )
    out["crp_mg_dL"] = out["crp_ug_mL"] / 10.0
    out["lymphocyte_pct"] = pd.to_numeric(lab["X_P_LIN"], errors="coerce")
    out["mcv_fL"] = pd.to_numeric(lab["X_VGM"], errors="coerce")
    out["rdw_pct"] = pd.to_numeric(lab["X_IDE"], errors="coerce")
    out["alkaline_phosphatase_U_L"] = pd.to_numeric(lab["X_PALK"], errors="coerce")
    out["wbc_1000_per_uL"] = pd.to_numeric(lab["X_GB"], errors="coerce")
    out["ln_crp_mg_dL"] = np.nan
    positive_crp = out["crp_mg_dL"] > 0
    out.loc[positive_crp, "ln_crp_mg_dL"] = np.log(
        out.loc[positive_crp, "crp_mg_dL"]
    )

    range_report: list[dict[str, Any]] = []
    if range_filter:
        for column, (low, high) in PLAUSIBLE_RANGES.items():
            outside = out[column].notna() & ~out[column].between(low, high)
            range_report.append(
                {
                    "variable": column,
                    "low": low,
                    "high": high,
                    "n_set_missing": int(outside.sum()),
                }
            )
            out.loc[outside, column] = np.nan

    terms = [name for name in COEFFICIENTS if name != "intercept"]
    complete = out.loc[:, terms].notna().all(axis=1)
    xb = np.full(len(out), np.nan, dtype=float)
    xb_complete = np.full(int(complete.sum()), COEFFICIENTS["intercept"], dtype=float)
    for term in terms:
        xb_complete += COEFFICIENTS[term] * out.loc[complete, term].to_numpy(float)
    xb[complete.to_numpy()] = xb_complete
    cumulative_hazard = np.full(len(out), np.nan, dtype=float)
    cumulative_hazard[complete.to_numpy()] = np.exp(np.clip(xb_complete, -745, 700)) * (
        np.expm1(GAMMA_PER_MONTH * FOLLOWUP_MONTHS) / GAMMA_PER_MONTH
    )
    out["clinical_phenoage_xb"] = xb
    out["clinical_phenoage_10yr_mortality_score"] = 1.0 - np.exp(
        -np.clip(cumulative_hazard, 0.0, 745.0)
    )
    out["clinical_phenoage"] = AGE_CONVERSION_OFFSET + np.log(
        AGE_CONVERSION_MULTIPLIER * cumulative_hazard
    ) / AGE_CONVERSION_SCALE
    out["clinical_phenoage_accel_raw"] = out["clinical_phenoage"] - out["age_years"]
    out["clinical_phenoage_complete_case"] = complete
    out = out.dropna(subset=["ID"]).copy()
    if out["ID"].duplicated().any():
        raise ValueError(f"{path} has duplicate baseline laboratory rows for an ID")
    return out, range_report


def load_vital_table(path: Path) -> pd.DataFrame:
    """Load unique participant mortality status, wave, and follow-up dates."""
    vital = pd.read_sas(path, format="sas7bdat", encoding="latin1")
    vital.columns = [str(column) for column in vital.columns]
    required = ("CODE98", "DECEASED", "INTV_MOR", "DATA_MOR", "DATA_ULT")
    require_columns(vital, set(required), path)
    optional = [column for column in ("ANNO_MOR", "STAT_ULT") if column in vital]
    out = vital.loc[:, [*required, *optional]].rename(columns={"CODE98": "ID"}).copy()
    out["ID"] = pd.to_numeric(out["ID"], errors="coerce")
    out["DECEASED"] = pd.to_numeric(out["DECEASED"], errors="coerce")
    out["INTV_MOR"] = pd.to_numeric(out["INTV_MOR"], errors="coerce")
    out = out.dropna(subset=["ID"]).copy()
    if out["ID"].duplicated().any():
        raise ValueError(f"{path} has duplicate vital-status rows for an ID")
    return out


def build_metadata(
    pheno: pd.DataFrame,
    beta_sample_ids: pd.Index,
    clinical: pd.DataFrame,
    vital: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Join phenotype, clinical, and vital data in beta-matrix sample order.

    Returns the sample metadata and the corresponding integer beta-matrix
    column positions. Survival and wave-based Kaplan-Meier endpoints are
    derived here from the joined vital-status fields.
    """
    keep = [
        column for column in ("ID", "sampleID", "visit", "AGE", "SEX") if column in pheno
    ]
    metadata = pheno.loc[:, keep].rename(
        columns={"visit": "dnam_visit", "AGE": "dnam_age", "SEX": "dnam_sex"}
    )
    metadata = metadata[metadata["sampleID"].isin(beta_sample_ids)].copy()
    metadata["n_dnam_samples"] = metadata.groupby("ID")["sampleID"].transform("size")
    metadata = metadata.merge(clinical, on="ID", how="left", validate="many_to_one")
    metadata = metadata.merge(vital, on="ID", how="left", validate="many_to_one")

    deceased = pd.to_numeric(metadata["DECEASED"], errors="coerce")
    metadata["event"] = np.where(deceased.notna(), deceased.eq(1).astype(float), np.nan)
    death_date = sas_date(metadata["DATA_MOR"])
    last_date = sas_date(metadata["DATA_ULT"])
    endpoint = death_date.where(metadata["event"].eq(1), last_date)
    metadata["survival_time_years"] = (
        endpoint - pd.to_datetime(metadata["lab_date"], errors="coerce")
    ).dt.days / 365.25
    clinical_sex = pd.to_numeric(metadata["SEX"], errors="coerce")
    if "dnam_sex" in metadata:
        dnam_sex = pd.to_numeric(metadata["dnam_sex"], errors="coerce")
    else:
        dnam_sex = pd.Series(np.nan, index=metadata.index)
    combined_sex = clinical_sex.combine_first(dnam_sex)
    metadata["sex_male"] = np.where(
        combined_sex.notna(), combined_sex.eq(1).astype(float), np.nan
    )

    wave = pd.to_numeric(metadata["INTV_MOR"], errors="coerce")
    death_wave = wave.isin([4, 5, 6])
    alive_wave = wave.eq(0)
    metadata["km_event"] = np.where(
        death_wave, 1.0, np.where(alive_wave, 0.0, np.nan)
    )
    metadata["km_time"] = np.where(
        death_wave, wave, np.where(alive_wave, 6.0, np.nan)
    )

    sample_position = pd.Series(np.arange(len(beta_sample_ids)), index=beta_sample_ids)
    positions = metadata["sampleID"].map(sample_position)
    if positions.isna().any():
        missing = metadata.loc[positions.isna(), "sampleID"].tolist()
        raise ValueError(f"pheno.baseline sample IDs absent from b.baseline: {missing[:5]}")
    metadata["clinical_phenoage_complete_case"] = metadata[
        "clinical_phenoage_complete_case"
    ].fillna(False).astype(bool)
    return metadata.reset_index(drop=True), positions.to_numpy(dtype=int)


def write_dataset(
    output: Path,
    metadata: pd.DataFrame,
    beta: Any,
    cpg_names: pd.Index,
    sample_positions: np.ndarray,
) -> int:
    """Validate the DNAm manifest and write the row-oriented analysis dataset.

    The return value is the number of DNAm features written.
    """
    selected_names = cpg_names.astype(str)
    prefix_counts = {
        prefix: int(selected_names.str.startswith(prefix).sum())
        for prefix in DNAM_FEATURE_PREFIXES
    }
    if (
        len(selected_names) != INCHIANTI_DNAM_FEATURE_COUNT
        or prefix_counts != DNAM_FEATURE_PREFIX_COUNTS
    ):
        raise ValueError(
            "b.baseline does not match the Figure 4 DNAm feature manifest: "
            f"n={len(selected_names)}, prefix_counts={prefix_counts}; expected "
            f"n={INCHIANTI_DNAM_FEATURE_COUNT}, "
            f"prefix_counts={DNAM_FEATURE_PREFIX_COUNTS}."
        )
    matrix = np.asarray(
        beta.values[:, sample_positions].T,
        dtype=np.float32,
    )
    beta_frame = pd.DataFrame(matrix, columns=selected_names)
    wide = pd.concat([metadata.reset_index(drop=True), beta_frame], axis=1, copy=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffixes = [suffix.lower() for suffix in output.suffixes]
    if output.suffix.lower() in {".parquet", ".pq"}:
        wide.to_parquet(output, index=False)
    elif ".csv" in suffixes:
        wide.to_csv(output, index=False)
    else:
        raise ValueError("--output must end in .parquet, .pq, .csv, or .csv.gz")
    return len(selected_names)


def main() -> None:
    """Prepare the shared table and record external input paths and software versions."""
    beta, cpg_names, beta_sample_ids, pheno = load_rdata(DNAM_RDATA_PATH)
    clinical, range_report = build_clinical_table(
        LAB_SAS_PATH, CRP_SOURCE, APPLY_PLAUSIBLE_RANGE_FILTER
    )
    vital = load_vital_table(VITAL_SAS_PATH)
    metadata, sample_positions = build_metadata(pheno, beta_sample_ids, clinical, vital)
    n_dnam_features = write_dataset(OUTPUT_PATH, metadata, beta, cpg_names, sample_positions)
    summary = {
        "inputs": {
            "dnam_rdata": str(DNAM_RDATA_PATH.resolve()),
            "lab_sas": str(LAB_SAS_PATH.resolve()),
            "vital_sas": str(VITAL_SAS_PATH.resolve()),
        },
        "output": relative_path(OUTPUT_PATH),
        "rdata_objects": ["b.baseline", "pheno.baseline"],
        "n_samples": len(metadata),
        "n_participants": int(metadata["ID"].nunique()),
        "n_duplicate_participants": int(
            metadata.loc[metadata["n_dnam_samples"].gt(1), "ID"].nunique()
        ),
        "n_clinical_phenoage_complete": int(
            metadata["clinical_phenoage_complete_case"].sum()
        ),
        "n_survival_events": int(metadata["event"].eq(1).sum()),
        "n_dnam_features": n_dnam_features,
        "dnam_feature_prefix_counts": DNAM_FEATURE_PREFIX_COUNTS,
        "dnam_feature_policy": "all b.baseline rows in original RData order",
        "crp_source": CRP_SOURCE,
        "range_filter": APPLY_PLAUSIBLE_RANGE_FILTER,
        "range_filter_report": range_report,
        "clinical_phenoage_formula": {
            "source": FORMULA_SOURCE,
            "coefficients": COEFFICIENTS,
            "gamma_per_month": GAMMA_PER_MONTH,
            "followup_months": FOLLOWUP_MONTHS,
            "age_conversion_offset": AGE_CONVERSION_OFFSET,
            "age_conversion_scale": AGE_CONVERSION_SCALE,
            "age_conversion_multiplier": AGE_CONVERSION_MULTIPLIER,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
            "rdata": version("rdata"),
        },
    }
    summary_path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.name}.metadata.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(metadata)} samples and {n_dnam_features} DNAm features to {OUTPUT_PATH}"
    )
    print(f"Wrote metadata to {summary_path}")


if __name__ == "__main__":
    main()
