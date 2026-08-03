# Figure 3 datasets

The two small TSV inputs are retained in the repository.  The ten prediction
parquets are not copied because they are model outputs rather than compact
manuscript tables.

Expected layout:

```text
fig3/datasets/
  pairwise_evolutionary_distances.tsv
  species_life_history.tsv
  loo_predictions/
    loo_per_species_predictions_BosTaurus.parquet
    ... one file for each of the ten species
```

Each prediction parquet must contain `age` plus nine columns named
`<component_species>_elasticnetcv`.  Replace the contents of
`fig3/datasets/loo_predictions`, or pass another project-relative directory
with `--loo-dir fig3/datasets/loo_predictions`.

`species_life_history.tsv` records maximum lifespan in years and gestation in
days.  Values were extracted from sheet `S1.13_anAge` of the corrected Lu et
al. (2023) Supplementary Data 1–14 workbook (Nature Aging,
doi:10.1038/s43587-023-00462-6).
