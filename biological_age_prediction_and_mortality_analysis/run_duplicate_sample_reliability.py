#!/usr/bin/env python3
"""Evaluate duplicate samples with the same held-out OOF models as panel 1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

from .config import INCHIANTI_DUPLICATE_PAIRS
from .data import complete_case_mask, read_analysis_dataset
from .plotting import COLORS, LABELS, clean_axis, configure_style, save_vector_pair
from .provenance import software_versions

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "outputs" / "preprocessing" / "inchianti_analysis_ready.parquet"
PREDICTIONS_PATH = ROOT / "outputs" / "biological_age_prediction" / "oof_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "duplicate_sample_reliability"

TARGET = "clinical_phenoage"
COMPLETE_CASE = "clinical_phenoage_complete_case"
MODEL_ORDER = ("bio_metaclock", "baseline")
DATASET_COLUMNS = ["ID", "sampleID", "age_years", TARGET, COMPLETE_CASE]


def valid_sample_manifest(path: Path) -> pd.DataFrame:
    """Return the exact Clinical-PhenoAge-complete DNAm sample manifest."""
    frame = read_analysis_dataset(path, columns=DATASET_COLUMNS)
    frame["ID"] = pd.to_numeric(frame["ID"], errors="coerce")
    frame["sampleID"] = frame["sampleID"].astype(str)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.loc[complete_case_mask(frame[COMPLETE_CASE])].dropna(
        subset=["ID", "sampleID", TARGET]
    )
    if frame["sampleID"].duplicated().any():
        raise ValueError("sampleID must identify one DNAm row.")
    return frame.reset_index(drop=True)


def load_formal_oof(path: Path, valid: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the two trained-model predictions from panel 1."""
    if not path.exists():
        raise FileNotFoundError(
            f"Run run_biological_age_prediction.py first; missing OOF predictions: {path}"
        )
    predictions = pd.read_csv(path)
    required = {"ID", "sampleID", "fold", "model", "pred_bioage"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"OOF predictions are missing columns: {sorted(missing)}")
    predictions = predictions[predictions["model"].isin(MODEL_ORDER)].copy()
    predictions["ID"] = pd.to_numeric(predictions["ID"], errors="coerce")
    predictions["sampleID"] = predictions["sampleID"].astype(str)
    expected = pd.MultiIndex.from_product(
        [valid["sampleID"].tolist(), MODEL_ORDER], names=["sampleID", "model"]
    )
    observed = pd.MultiIndex.from_frame(predictions[["sampleID", "model"]])
    if observed.has_duplicates or set(observed) != set(expected):
        raise ValueError("OOF predictions must contain one row per valid sample and trained model.")
    id_by_sample = valid.set_index("sampleID")["ID"]
    expected_id = predictions["sampleID"].map(id_by_sample)
    if not predictions["ID"].eq(expected_id).all():
        raise ValueError("OOF prediction IDs do not match the valid sample manifest.")
    if predictions.groupby("ID")["fold"].nunique().max() != 1:
        raise ValueError("Both DNAm rows from a participant must use the same held-out fold.")
    return predictions


def paired_differences(predictions: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Collapse the valid exactly-two-sample participants into paired differences."""
    counts = valid.groupby("ID")["sampleID"].nunique()
    duplicate_ids = set(counts[counts.eq(2)].index)
    held_out = predictions[predictions["ID"].isin(duplicate_ids)].copy()
    rows: list[dict[str, object]] = []
    for (participant, model), pair in held_out.groupby(["ID", "model"], sort=True):
        pair = pair.sort_values("sampleID", kind="mergesort")
        if len(pair) != 2:
            raise ValueError(f"Expected two valid samples for ID={participant}, model={model}.")
        first, second = pair.iloc[0], pair.iloc[1]
        prediction_1 = float(first["pred_bioage"])
        prediction_2 = float(second["pred_bioage"])
        rows.append(
            {
                "ID": participant,
                "model": model,
                "fold": int(first["fold"]),
                "sampleID_1": first["sampleID"],
                "sampleID_2": second["sampleID"],
                "prediction_1": prediction_1,
                "prediction_2": prediction_2,
                "absolute_delta": abs(prediction_1 - prediction_2),
            }
        )
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Calculate the exact Figure 4 duplicate-reliability source columns."""
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        subset = pairs[pairs["model"].eq(model)]
        rows.append(
            {
                "model": model,
                "n_pairs": len(subset),
                "pearson_r": float(
                    pearsonr(subset["prediction_1"], subset["prediction_2"]).statistic
                ),
                "mean_absolute_delta": float(subset["absolute_delta"].mean()),
                "median_absolute_delta": float(subset["absolute_delta"].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(summary: pd.DataFrame) -> None:
    """Draw the two-model Figure 4 duplicate panel."""
    configure_style()
    table = summary.set_index("model").loc[list(MODEL_ORDER)]
    figure, axis = plt.subplots(figsize=(60 / 25.4, 50 / 25.4))
    axis.bar(
        [LABELS[model] for model in MODEL_ORDER],
        table["mean_absolute_delta"],
        color=[COLORS[model] for model in MODEL_ORDER],
        edgecolor="#333333",
        linewidth=0.2,
    )
    axis.set(ylabel="Mean |Δ| (years)", title="Variability Across Replicates")
    clean_axis(axis)
    figure.tight_layout()
    save_vector_pair(figure, OUTPUT_DIR, "02_duplicate_sample_reliability_barplot")


def main() -> int:
    """Reuse formal OOF predictions and summarize the valid replicate pairs."""
    valid = valid_sample_manifest(DATASET_PATH)
    predictions = load_formal_oof(PREDICTIONS_PATH, valid)
    pairs = paired_differences(predictions, valid)
    if pairs["ID"].nunique() != INCHIANTI_DUPLICATE_PAIRS:
        raise ValueError(
            f"Figure 4 duplicate analysis requires {INCHIANTI_DUPLICATE_PAIRS} valid pairs."
        )
    summary = summarize_pairs(pairs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT_DIR / "duplicate_pair_absolute_deltas.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "duplicate_sample_reliability.csv", index=False)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "prediction_source": str(PREDICTIONS_PATH.resolve()),
                "complete_case_filter": COMPLETE_CASE,
                "pair_rule": "exactly two complete-case DNAm samples per participant",
                "held_out_model_rule": "same participant-grouped OOF model as panel 1",
                "models": list(MODEL_ORDER),
                "n_pairs": int(pairs["ID"].nunique()),
                "software": software_versions(["pandas", "scipy"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_reliability(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
