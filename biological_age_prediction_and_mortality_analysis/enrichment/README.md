# CpG-weight enrichment workflow

This subworkflow runs the top/bottom-1,000 GREAT analysis from the published
CpG selection table.

## Published foreground interface

`select_top_bottom_cpg_foregrounds.py` reads
`top_bottom_1000_cpg_weights.csv` directly. It validates the four published
model/direction groups, joins their CpG identifiers to the array-union hg19
coordinates, and writes the BED foregrounds used by rGREAT. Component-level
coefficient files and full-ranking audit files are not required.

## Array and mapping audits

`build_27k_450k_epic_union_background.R` uses pinned maintained Bioconductor
annotation packages. The 27K v1.2 annotation is lifted from hg18 to hg19 with a
user-supplied UCSC chain through `rtracklayer`


## Commands

From the project root:

```bash
Rscript biological_age_prediction_and_mortality_analysis/enrichment/build_27k_450k_epic_union_background.R results/enrichment/array_union

python -m biological_age_prediction_and_mortality_analysis.enrichment.select_top_bottom_cpg_foregrounds

for library in go_bp kegg_2021_human reactome_pathways_2024 msigdb_hallmark_2020; do
  for model in chrono bio; do
    Rscript biological_age_prediction_and_mortality_analysis/enrichment/run_rgreat_top_bottom_enrichment.R "$model" "$library"
  done
done

python -m biological_age_prediction_and_mortality_analysis.enrichment.plot_top_bottom_enrichment
```
