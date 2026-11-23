# Intervention and exposure analyses

Figure 5 uses three separately tested screens. The included tables contain
group-level results and the Benjamini–Hochberg adjusted P values for each screen.

| Screen | Model | Comparisons | BH FDR ≤ 0.05 |
| --- | --- | ---: | ---: |
| Framingham Heart Study (FHS) | MetaClock-5 | 433 | 109 |
| Women's Health Initiative (WHI) | MetaClock-19 | 428 | 46 |
| Filtered external GEO and TranslAGE studies | MetaClock-19 | 68 | 4 |

FHS and WHI compare covariate-adjusted scores between exposure groups using
two-sided Welch tests. The external screen includes 68 comparisons from 54
accessions. Of the 94 initial nonredundant comparisons, 24 lacked recoverable
individual ages, one had unresolved donor pairs, and GSE144804 provided only
a newborn age category. The retained set contains 34 paired and 34 Welch tests.
BH correction uses all 68 retained comparisons as one family.
The source screens used Python 3.11.14, SciPy 1.17.0 and statsmodels 0.14.6.

## Figure 5 volcano plots

From the repository root, render all three panels with one command.

```bash
python -m intervention_analysis.run_biometaclock_intervention_analysis
```

The command uses the bundled FHS and WHI results and recomputes the external
effects, two-sided P values and BH correction from donor-level summary statistics.
The external table supplies each arm’s donor count, mean and sample standard
deviation, plus the SD of within-donor differences for paired tests. Scores are
age-adjusted. For parallel-arm longitudinal studies the arm means represent
donor-level changes. The eligibility ledger identifies the retained comparisons
and gives the reason for each exclusion.

Outputs under `results/figure5/` include `combined_geo_translage_results_bh.csv`,
`external_analysis_summary.json` and the three PDF/SVG panels with source audits.
The external panel labels every comparison at BH FDR ≤ 0.05. The FHS and WHI
panels label up to five positive and five negative effects meeting that cutoff.
The individual-panel CLI accepts a supplied results CSV with
`--cohort fhs`, `--cohort whi`, or `--combined-geo-translage`.

## Re-score the external model

To start from the original 19 component predictions and sample design, supply
the prediction run’s task manifests and shards. The scorer applies the frozen
19-component InCHIANTI model, fits age residualization within the original
dataset strata before selecting contrasts, collapses technical replicates by
donor and recomputes the paired or Welch comparisons. The two GrimAge inputs
are the component outputs calculated using the recorded age and recorded or
inferred sex. This entry point starts after component-clock calculation.

```bash
python -m intervention_analysis.run_biometaclock_intervention_analysis \
  --run-dir path/to/your/metaclock19_prediction_run \
  --output-dir results/figure5_rescored
```

## External intervention UMAP

The embedding uses the same bundled external comparisons. Supply the corresponding
MetaClock-19 prediction run directory.
The run directory contains `tasks_initial.json`, `tasks_recovered.json`, and the
`shards/` sample, donor, and component prediction tables.

```bash
uv sync --frozen --group intervention-umap
uv run --frozen --group intervention-umap python -m intervention_analysis.external_umap \
  --run-dir path/to/your/metaclock19_prediction_run \
  --output-dir results/intervention/external_umap
```

Each tested sample enters the embedding once. A sample in both arms across
comparisons takes the intervention color. The embedding uses all 19 component predictions,
including GrimAge and GrimAge2, before age residualization. Each component is
standardized across the included samples. The two-dimensional Euclidean UMAP
uses 15 neighbors, `min_dist=1`, `spread=10`, and seed 42. Outputs include the
PDF, SVG, PNG, interactive HTML, coordinates, comparison memberships, fitted
scaler and UMAP, and a run summary.

## External candidate screen

The upstream candidate workflow applies the frozen GEO and TranslAGE screening
rules. It starts with 77 GEO comparisons and 19 TranslAGE comparisons.
Cross-source deduplication yields a preliminary family of 94 comparisons from
75 accessions. Its bundled weights are the published 17-component clock.
Figure 5 uses the filtered MetaClock-19 results described above.

```bash
python -m intervention_analysis.fetch_geo_datasets_and_metadata
python -m intervention_analysis.fetch_translage_datasets catalog
python -m intervention_analysis.screen_translage_datasets
python -m intervention_analysis.fetch_translage_datasets matrices
python -m intervention_analysis.run_external_candidate_screen
```

Candidate results are written to `results/intervention/biometaclock_94/`.
Use `--workers` to set the number of prediction processes (allow up to
64 GiB each), `--only-gse` for a subset, or `--force` to rerun cached
predictions. A subset run skips the combined BH correction and plot.
