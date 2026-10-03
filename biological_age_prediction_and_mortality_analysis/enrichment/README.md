# CpG-weight enrichment workflow

This subworkflow runs the top/bottom-1,000 GREAT analysis from the published
CpG selection table.

## Array and mapping audits

`build_27k_450k_epic_union_background.R` uses pinned maintained Bioconductor
annotation packages. The 27K v1.2 annotation is lifted from hg18 to hg19 with a
user-supplied UCSC hg18-to-hg19 chain through `rtracklayer`. Replace
`path/to/your/file` in the command below with this chain file, supplied as the
second positional argument. The script uses the same placeholder as its default.

External gene-set libraries are read from `<library>.gmt` files. Replace
`path/to/your/directory` below with the directory containing
`kegg_2021_human.gmt`, `reactome_pathways_2024.gmt`, and
`msigdb_hallmark_2020.gmt`.


## Commands

From the project root:

```bash
Rscript biological_age_prediction_and_mortality_analysis/enrichment/build_27k_450k_epic_union_background.R results/enrichment/array_union path/to/your/file

python -m biological_age_prediction_and_mortality_analysis.enrichment.select_top_bottom_cpg_foregrounds

for model in chrono bio; do
  Rscript biological_age_prediction_and_mortality_analysis/enrichment/run_rgreat_top_bottom_enrichment.R "$model" go_bp
done

for library in kegg_2021_human reactome_pathways_2024 msigdb_hallmark_2020; do
  for model in chrono bio; do
    Rscript biological_age_prediction_and_mortality_analysis/enrichment/run_rgreat_top_bottom_enrichment.R "$model" "$library" \
      results/enrichment/inputs \
      results/enrichment/array_union/background_27k_450k_epic_union_hg19.bed.gz \
      path/to/your/directory
  done
done

python -m biological_age_prediction_and_mortality_analysis.enrichment.plot_top_bottom_enrichment
```
