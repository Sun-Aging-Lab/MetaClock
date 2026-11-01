# GEO and TranslAGE intervention analysis

Combines 77 comparisons from 64 GEO series with 19 TranslAGE comparisons from
13 public accessions. Removing cross-source overlaps leaves 94 tests from
75 accessions, with BH correction across all 94 tests.

## Run

Run these commands in order using the `metaclock` conda environment:

```bash
cd /orcd/pool/007/edsun_shared/yncheng/metaclock

conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.fetch_geo_datasets_and_metadata
conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.fetch_translage_datasets catalog
conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.screen_translage_datasets
conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.fetch_translage_datasets matrices
conda run --no-capture-output -n metaclock \
  python -m intervention_analysis.run_biometaclock_intervention_analysis
```

Results are written to
`/orcd/pool/007/edsun_shared/yncheng/metaclock/results/intervention/biometaclock_94`.


## Analysis

TranslAGE is limited
to the 19 reviewed longitudinal Intervention/Event comparisons; new catalog
entries do not expand the analysis.

TranslAGE averages consistent technical replicates and uses complete donor pairs
at their earliest and latest visits (42 days for GSE191297). Single-arm studies
use paired t-tests; parallel-arm studies compare donor changes using Welch tests.
Within-dataset linear age adjustment is applied when chronological ages are
complete. CpG coverage, beta-value, and donor-count checks remain in place.

## Options and caching

- `--workers`: concurrent prediction processes; allow up to 64 GiB per process.
- `--only-gse`: run selected GEO series for debugging; subsets skip final BH
  correction and plotting.
- `--force`: rerun cached predictions.

Dataset consistency is checked by IDs and public accessions. Cache reuse checks
comparison/sample IDs, groups, model parameters, required outputs, and input
modification times. Use `--force` if replacing inputs while preserving their old
modification times.
