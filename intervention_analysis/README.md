# GEO and TranslAGE intervention analysis

The frozen workflow combines 77 comparisons from 64 GEO series with 19
TranslAGE comparisons from 13 public accessions. GSE201532 and GSE74548 occur
in both sources: retain the audited TranslAGE designs and remove the two older
GEO results. The final family contains 94 tests from 75 accessions.

Run commands from the repository root. CLI paths are resolved relative to the
repository root, not the current working directory; absolute paths and paths
that leave the repository are rejected.

## Fetch and screen

Fetch the original GEO matrices and metadata:

```bash
conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.fetch_geo_datasets_and_metadata
```

Fetch the TranslAGE catalog, screen it, then download the selected cleaned RData:

```bash
conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.fetch_translage_datasets catalog
conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.screen_translage_datasets
conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.fetch_translage_datasets matrices
```

The catalog screen requires longitudinal Intervention/Event records assigned to
TranslAGE's Responsive analysis and at least four reported subjects or controls.
The frozen 2026-08-26 snapshot contains 179 records, 95 candidates, and 23 reviewed
official cleaned-data entries. Selection removes GSE191297's two-week entry in
favor of six weeks and the three pooled E-MTAB-8956 entries in favor of four
disjoint diet-by-exercise arms, leaving 19 comparisons. Prior-GEO overlap is
marked during screening and resolved when combining test results.

The catalog, screening audit, and analysis manifest are written under
`data/intervention/translage/catalog`; cleaned matrices go under
`data/intervention/translage/official`. The two tracked files
`config/translage_cleaned_inventory.csv` and `config/translage_provenance.json`
record the reviewed URLs, source hashes, sizes, and reference snapshot. Downloads
must match those hashes; changed catalog or payload content requires explicit
re-curation. No alternative data source is selected automatically.

## Predict, combine, and plot

```bash
conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.run_biometaclock_intervention_analysis
```

The main script runs the existing GEO adapter and the repository-local
`run_translage_biometaclock.py` adapter. TranslAGE uses curated sample, donor,
arm, age, and time fields. Consistent technical replicates are averaged;
samples without intervention times are excluded. Complete donor pairs use their
earliest and latest visits (42 days for GSE191297). Single-arm studies use paired
t-tests; parallel-arm studies compare donor endpoint-minus-baseline changes with
Welch tests. Complete chronological ages trigger the existing within-dataset
linear residualization. CpG coverage, beta-distribution, and donor-count gates
match the original analysis.

Both sources use the same 17-component biological MetaClock reconstructed from
`biological_age_prediction_and_mortality_analysis/model_weights/`. PyAging 0.1.30
uses the repository-local `data/intervention/pyaging_cache` for its component
models; first use downloads about 22.5 GiB. The frozen GEO matrices occupy about
28.7 GiB. Package versions are pinned in `pyproject.toml`, and run summaries
record the actual versions, model hashes, manifest hashes, and QC parameters.

Results are written to `results/intervention/biometaclock_94`. The main script
checks exact sample-ID overlap across the 77-by-19 comparisons, retains
TranslAGE for the two shared accessions, and applies statsmodels BH across all
94 raw p-values, independently checked against SciPy. Source-family q-values
are retained only as provenance. Outputs include:

- `results_77_interventions.csv` and `results_19_translage_interventions.csv`;
- `overlap_resolution_audit.csv` and `cross_source_sample_overlap_audit.csv`;
- `combined_geo_translage_results_bh.csv`, strict and exploratory result tables;
- `volcano/combined_geo_translage_biometaclock_volcano_fdr_5_20.pdf` and `.svg`,
  plus the plot's JSON audit.

The reference combined plot has four tests at FDR ≤ 5% and three at
5% < FDR ≤ 20%; all seven are labeled. The plot uses combined BH values and
preserves the original GEO axis limits. The plotting script's explicit
`--combined-geo-translage` mode reads this schema; its GEO-only mode remains
available for the historical 77-test result.

`--workers` controls concurrent prediction processes. Each process may require
64 GiB; select workers to fit the allocated memory. `--only-gse` runs a GEO
subset for debugging without producing an incomplete final BH family.
`--force` reruns cached prediction shards. Malformed cache files raise an error;
only absent or outdated caches are recomputed automatically.
