"""Fit and plot an intervention UMAP from tested donor and component-prediction tables."""

from __future__ import annotations

import argparse
import hashlib
import html
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from biological_age_prediction_and_mortality_analysis.config import MANIFEST_PATH, OOF_COMPONENTS
from intervention_analysis.external_umap_plot import COLORS, STEM, render
from intervention_analysis.paths import PROJECT_ROOT


PARAMETERS = dict(n_components=2, n_neighbors=15, min_dist=1.0, spread=10.0, metric="euclidean",
                  random_state=42, transform_seed=42, n_jobs=1, init="spectral", low_memory=True)


def tested_members(meta: pd.Series, samples: pd.DataFrame, donors: pd.DataFrame) -> pd.DataFrame:
    keys = ["group", "donor_id"]
    selected = samples.loc[samples.comparison_id.eq(meta.comparison_id)].copy()
    tested = donors.loc[donors.comparison_id.eq(meta.comparison_id)]
    signature = "sample_signature_sha256" if meta.sample_signature_sha256 else "sample_signature_id"
    for column in ("analysis_method", "gse_id", signature):
        if set(selected[column]) != {meta[column]}:
            raise ValueError(f"{meta.name}: mismatched {column}")
    selected = selected.merge(tested[keys], on=keys, validate="many_to_one")
    if not selected.sample_id.is_unique or selected.sample_id.eq("").any():
        raise ValueError(f"{meta.name}: missing or repeated sample ID")
    counts = selected.groupby(keys).size().sort_index()
    expected = tested.set_index(keys).n_samples_collapsed.astype(int).sort_index()
    if not counts.equals(expected):
        raise ValueError(f"{meta.name}: tested donor membership differs from donor predictions")
    arm_sizes = {"control": int(meta.control_n_donors), "experiment": int(meta.experiment_n_donors)}
    if selected.groupby("group").donor_id.nunique().to_dict() != arm_sizes:
        raise ValueError(f"{meta.name}: tested arm sizes differ from panel")
    if meta.test == "two_sided_paired_t_test" and not selected.groupby("donor_id").group.nunique().eq(2).all():
        raise ValueError(f"{meta.name}: incomplete donor pairs")
    selected["family_test_id"] = meta.name
    selected["public_id"] = meta.public_id
    selected["intervention"] = meta.display_intervention
    if "timepoint" not in selected:
        selected["timepoint"] = "comparison_group"
    return selected[["sample_id", "family_test_id", "public_id", "intervention", "group",
                     "donor_id", "timepoint", "chronological_age"]]


def assemble(panel: pd.DataFrame, run: Path, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    memberships, clock_rows, sources = [], [], []
    represented: set[str] = set()
    for name in ("tasks_initial.json", "tasks_recovered.json"):
        manifest = run / name
        sources.append(manifest)
        for task in json.loads(manifest.read_text()):
            families = panel.index.intersection(task["retained_families"])
            if families.empty:
                continue
            if represented.intersection(families):
                raise ValueError(f"Multiple shards for a panel comparison in {task['task_key']}")
            represented.update(families)
            if set(panel.loc[families, "public_id"]) != {task["public_id"]}:
                raise ValueError(f"Panel accession differs from shard {task['task_key']}")
            shard = run / "shards" / task["task_key"]
            sample_file = shard / "analysis_sample_predictions.csv"
            donor_file = shard / "donor_predictions.csv"
            clock_file = shard / "component_predictions_19.csv"
            sources.extend((sample_file, donor_file, clock_file))
            samples = pd.read_csv(sample_file, dtype=str, keep_default_na=False)
            donors = pd.read_csv(donor_file, dtype=str, keep_default_na=False)
            members = pd.concat([tested_members(panel.loc[family], samples, donors) for family in families])
            memberships.append(members)
            clocks = pd.read_csv(clock_file, usecols=["sample_id", *features], dtype={"sample_id": str})
            clocks = clocks.set_index("sample_id", verify_integrity=True)
            clock_rows.append(clocks.loc[sorted(set(members.sample_id)), features])
    if not represented or represented != set(panel.index):
        raise ValueError(f"Missing panel comparisons: {sorted(set(panel.index) - represented)}")
    members = pd.concat(memberships, ignore_index=True).sort_values(["sample_id", "family_test_id"])
    clocks = pd.concat(clock_rows).sort_index(kind="stable")
    unique = clocks.loc[~clocks.index.duplicated()]
    if not np.isfinite(clocks).all().all():
        raise ValueError("Nonfinite component predictions")
    if not np.allclose(clocks, unique.loc[clocks.index], rtol=1e-7, atol=1e-7):
        raise ValueError("Shared sample IDs have conflicting component predictions")
    return members, unique, sources


def summarize_samples(members: pd.DataFrame) -> pd.DataFrame:
    rows = members.copy()
    rows["hover"] = (rows.public_id.map(html.escape) + " · " + rows.intervention.map(html.escape)
                     + " — " + rows.group + " (" + rows.timepoint.map(html.escape) + ")")
    by_sample = rows.groupby("sample_id", sort=True)
    samples = by_sample.agg(family_test_ids=("family_test_id", ";".join),
                            public_ids=("public_id", lambda values: ";".join(sorted(set(values)))),
                            comparison_memberships=("family_test_id", "nunique"),
                            role=("group", "first"), hover_labels=("hover", "<br>".join))
    samples.loc[by_sample.group.nunique().gt(1), "role"] = "both"
    samples["plot_group"] = np.where(samples.role.eq("control"), "control", "intervention")
    samples["marker_area_all"] = (np.where(samples.role.eq("control"), 7., 9.)
                                  * np.where(samples.comparison_memberships.gt(1), 1.3, 1.) / 4.)
    return samples


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_panel(results: Path, run: Path, output: Path) -> None:
    from umap import UMAP

    panel = pd.read_csv(results, dtype=str, keep_default_na=False).set_index("family_test_id", verify_integrity=True)
    features = list(OOF_COMPONENTS)
    members, raw, sources = assemble(panel, run, features)
    samples = summarize_samples(members)
    if not samples.index.equals(raw.index):
        raise ValueError("Component predictions and sample memberships do not align")
    scaler = StandardScaler()
    reducer = UMAP(**PARAMETERS)
    embedding = reducer.fit_transform(scaler.fit_transform(raw))
    if not np.isfinite(embedding).all() or np.ptp(embedding, axis=0).min() <= 0:
        raise ValueError("Invalid UMAP coordinates")
    samples[["umap_1", "umap_2"]] = embedding
    samples = samples.reset_index()
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        dest = Path(tmp)
        raw.to_csv(dest / "component_clock_inputs.csv")
        members.to_csv(dest / "sample_comparison_memberships.csv", index=False)
        samples.to_csv(dest / f"{STEM}_coordinates.csv", index=False)
        joblib.dump(scaler, dest / "feature_scaler.joblib")
        joblib.dump(reducer, dest / "umap_model.joblib")
        render(samples, dest)
        summary = {
            "comparisons": len(panel), "public_accessions": panel.public_id.nunique(),
            "unique_samples": len(samples), "memberships": len(members),
            "both_roles": int(samples.role.eq("both").sum()), "groups": samples.plot_group.value_counts().to_dict(),
            "colors": COLORS, "feature_names": features, "parameters": PARAMETERS,
            "python": platform.python_version(),
            "versions": {name: version(name) for name in ("umap-learn", "scikit-learn", "numba", "pynndescent",
                                                         "numpy", "scipy", "pandas", "joblib", "matplotlib", "plotly")},
            "input_scores": "Component predictions before age residualization; z-scored across included samples",
            "results_sha256": sha256(results), "feature_metadata_sha256": sha256(MANIFEST_PATH),
            "input_sha256": {path.relative_to(run).as_posix(): sha256(path) for path in sources},
            "output_sha256": {path.name: sha256(path) for path in sorted(dest.iterdir())},
        }
        audit = dest / "run_summary.json"
        audit.write_text(json.dumps(summary, indent=2) + "\n")
        for name in summary["output_sha256"]:
            (dest / name).replace(output / name)
        audit.replace(output / audit.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "screen_summaries" / "external_metaclock19.csv",
                        help="Filtered external comparison CSV")
    parser.add_argument("--run-dir", type=Path, required=True, help="MetaClock-19 task manifests and shards")
    parser.add_argument("--output-dir", type=Path, default=Path("results/intervention/external_umap"))
    args = parser.parse_args()
    fit_panel(*(PROJECT_ROOT / path.expanduser() for path in (args.results, args.run_dir, args.output_dir)))


if __name__ == "__main__":
    main()
