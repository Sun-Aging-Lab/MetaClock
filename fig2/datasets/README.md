# Figure 2 datasets

The repository does not include large methylation matrices.  Replace each
placeholder directory in `manifest.tsv` with your dataset at the same relative
location, or pass its path with `--dataset-dir`.

Expected files for an array or tissue dataset:

```text
fig2/datasets/<group>/<dataset>/
  beta_age.parquet                    # rows=samples, columns=CpGs plus age
  component_clock_predictions.parquet # rows=samples, 16 clocks plus age
```

`beta_age.parquet` must contain a numeric `age` column and numeric CpG columns
whose names begin with `cg`.  `component_clock_predictions.parquet` must use
the same sample index and contain the fixed component-clock predictions plus
`age`.  Disease cases and technical controls must be filtered upstream.

For a species dataset, ages and MAEs are on the relative-age scale.  Its inputs
are deliberately separate from the array/tissue schema:

```text
fig2/datasets/species/BosTaurus/
  BosTaurus_fewshot_40k.parquet
  loo_per_species_predictions_BosTaurus.parquet
```

The first table contains `age` plus conserved CpGs.  The second contains `age`
plus per-species meta features, of which only columns ending in `_mlp_elu` are
used. The two tables must have the same sample index, row order, and age values
so their fixed shuffle and split select the same participants. Species inputs
do not undergo the human MI/mRMR feature-selection step.

The manuscript paths in `manifest.tsv` are deliberately project-relative, for
example `fig2/datasets/array/GSE234461`.  Change that row to your own dataset
path if you do not use this layout.
