# CpG-weight enrichment workflow

This subworkflow builds the top/bottom-1,000 GREAT analysis from
normalized MetaClock/component coefficient inputs. It does not
contain intervention or volcano analyses.

`top_bottom_1000_cpg_weights.csv` is the compact publication table for the
selected foregrounds. Its 4,000 rows record `group` (`chronological` or
`biological`), `direction`, deterministic `rank`, CpG identifier, and the
aggregate weight used for selection. Each group/direction contains exactly
1,000 unique CpGs.

## Fixed scientific definition

- The chronological manifest records all 16 candidate clocks. Eight pass the
  fixed-affine end-to-end audit and therefore enter the CpG ranking; the
  other eight remain documented but cannot be assigned fixed CpG coefficients.
- The biological manifest records 19 candidates. `systemsage` and
  `dunedinpace` are non-affine and cannot yield a fixed CpG coefficient;
  therefore the CpG ranking uses the remaining 17 affine components.
- Each CpG contribution is the raw-scale MetaClock coefficient multiplied by
  the raw fixed-affine component CpG coefficient, summed over components.
  Intercepts do not affect this ranking.
- Foregrounds are the 1,000 greatest and 1,000 smallest weights after filtering
  only to the union. They have no 450K-only restriction.
- The hg19 union has exactly 896,980 unique CpGs: 27K v1.2 (27,574), 450K v1.2
  (482,421), and EPIC v1.0 B4 (862,927). EPIC v2 is not included.
- rGREAT 2.8.0 uses `GREAT:hg19`, `basalPlusExt`, 5 kb upstream, 1 kb downstream,
  1 Mb extension, `exclude=NULL`, gene-set size >=5, and
  `min_region_hits=5`. BH adjustment is within model/direction/library.

## Normalized coefficient interface

Raw PyAging/component artifact discovery is environment-specific.
`build_component_weight_rankings.py` starts from this portable interface:

- `meta_component_weights.csv`: `model`, `clock`, `coefficient`; manifest
  order for the selected model.
- `component_cpg_coefficients.csv.gz`: `clock`, `cpg`, `coefficient`; unique
  clock/CpG rows on the raw input scale.
- `component_model_audit.csv`: `clock`, `fixed_affine`, `artifact_sha256`;
  unique component records with a valid 64-character SHA-256 digest.

The ranking summary binds every input and generated ranking/contribution table
to SHA-256 hashes. Foreground selection verifies this summary before proceeding.

## Array and mapping audits

`build_27k_450k_epic_union_background.R` uses pinned maintained Bioconductor
annotation packages. The 27K v1.2 annotation is lifted from hg18 to hg19 with a
user-supplied UCSC chain through `rtracklayer`; the audit records chain SHA-256,
source/mapped/unmapped counts, and coordinate concordance with probes shared by
450K before and after lifting. The 450K v1.2 and EPIC v1.0 B4 annotations are
already hg19. The workflow rejects post-lift cross-platform conflicts and
duplicate union coordinates, and hashes the manifest, membership, and BED
output.

For KEGG, Reactome, and Hallmark, `AnnotationDbi::select` with `org.Hs.eg.db`
maps SYMBOL to ENTREZID. Mappings are deduplicated per `(term, entrez_id)` before
rGREAT and both the complete mapping and per-term input/mapped counts are saved.
GO:BP uses rGREAT's native GO identifier route and records an intentionally empty
SYMBOL mapping file. Region-gene associations and R session information are
retained for every result.

## Commands

From the project root:

```bash
Rscript biological_age_prediction_and_mortality_analysis/enrichment/build_27k_450k_epic_union_background.R results/enrichment/array_union

python -m biological_age_prediction_and_mortality_analysis.enrichment.build_component_weight_rankings chrono
python -m biological_age_prediction_and_mortality_analysis.enrichment.build_component_weight_rankings bio
python -m biological_age_prediction_and_mortality_analysis.enrichment.select_top_bottom_cpg_foregrounds

for library in go_bp kegg_2021_human reactome_pathways_2024 msigdb_hallmark_2020; do
  for model in chrono bio; do
    Rscript biological_age_prediction_and_mortality_analysis/enrichment/run_rgreat_top_bottom_enrichment.R "$model" "$library"
  done
done

python -m biological_age_prediction_and_mortality_analysis.enrichment.plot_top_bottom_enrichment
```

The eight semantic vector stems follow
`enrichment_{chronological|biological}_{library}_top_bottom_1000`.

KEGG, Reactome, and Hallmark are version-pinned GMT snapshots in the configured external
gene-set directory. Record their own source/version checksums alongside the input
data. The R environment requires rGREAT 2.8.0 and the package versions pinned in
the array annotation audit.
