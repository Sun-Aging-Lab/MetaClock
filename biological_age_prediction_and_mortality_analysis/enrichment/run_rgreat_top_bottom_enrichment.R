#!/usr/bin/env Rscript

# Run restricted-background GREAT enrichment for one model and one library.
suppressPackageStartupMessages({
    library(AnnotationDbi)
    library(GenomicRanges)
    library(org.Hs.eg.db)
    library(rGREAT)
})
script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1L]]), mustWork = TRUE)
manifest_path <- file.path(dirname(dirname(script_path)), "analysis_manifest.json")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = TRUE)
config <- manifest$enrichment
if (!identical(as.character(utils::packageVersion("rGREAT")), config$rGREAT_version)) {
    stop("This workflow requires rGREAT ", config$rGREAT_version)
}

default_input_dir <- "results/enrichment/inputs"
default_background <- "results/enrichment/array_union/background_27k_450k_epic_union_hg19.bed.gz"
default_gene_sets <- "/path/to/your/file" # fill in your path
default_output_dir <- "results/enrichment/rgreat"
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
    stop("Usage: Rscript run_rgreat_top_bottom_enrichment.R MODEL LIBRARY [INPUT_DIR BACKGROUND_BED GENE_SET_DIR OUTPUT_DIR]")
}
model <- args[[1L]]
library_name <- args[[2L]]
input_dir <- if (length(args) >= 3L) args[[3L]] else default_input_dir
background_path <- if (length(args) >= 4L) args[[4L]] else default_background
gene_set_dir <- if (length(args) >= 5L) args[[5L]] else default_gene_sets
output_root <- if (length(args) >= 6L) args[[6L]] else default_output_dir

valid_models <- unname(config$models)
valid_libraries <- unname(config$libraries)
if (!model %in% valid_models) stop("MODEL must be chrono or bio")
if (!library_name %in% valid_libraries) stop("Unknown library: ", library_name)

foreground_regions_expected <- as.integer(config$top_n)
background_regions_expected <- as.integer(config$union_cpgs)
min_gene_set_size <- as.integer(config$min_gene_set_size)
min_region_hits <- as.integer(config$min_region_hits)

read_bed <- function(path) {
    table <- utils::read.delim(
        path,
        header = FALSE,
        col.names = c("chromosome", "start", "end", "cpg"),
        colClasses = c("character", "integer", "integer", "character"),
        stringsAsFactors = FALSE
    )
    if (!nrow(table) || anyNA(table) || anyDuplicated(table$cpg)) stop("Invalid BED: ", path)
    if (any(table$start < 0L) || any(table$end <= table$start)) stop("Invalid BED intervals: ", path)
    if (anyDuplicated(paste(table$chromosome, table$start, table$end, sep = ":"))) {
        stop("BED contains duplicate genomic coordinates: ", path)
    }
    table
}

to_granges <- function(table) {
    ranges <- GRanges(
        seqnames = table$chromosome,
        ranges = IRanges::IRanges(start = table$start + 1L, end = table$end)
    )
    names(ranges) <- table$cpg
    ranges
}

map_symbol_sets_to_entrez <- function(gene_sets, output_dir) {
    # AnnotationDbi performs the maintained SYMBOL-to-ENTREZID mapping.
    pairs <- unique(do.call(rbind, lapply(names(gene_sets), function(term) {
        data.frame(term = term, symbol = gene_sets[[term]], stringsAsFactors = FALSE)
    })))
    mapping <- suppressMessages(AnnotationDbi::select(
        org.Hs.eg.db,
        keys = unique(pairs$symbol),
        columns = "ENTREZID",
        keytype = "SYMBOL"
    ))
    names(mapping) <- c("symbol", "entrez_id")
    mapped <- merge(pairs, mapping, by = "symbol", all.x = TRUE, sort = FALSE)
    mapped$mapped <- !is.na(mapped$entrez_id) & nzchar(as.character(mapped$entrez_id))
    mapped <- unique(mapped[order(mapped$term, mapped$symbol, mapped$entrez_id), ])
    utils::write.csv(mapped, file.path(output_dir, "gene_set_symbol_entrez_mapping.csv"), row.names = FALSE, na = "")
    mapped_pairs <- unique(mapped[mapped$mapped, c("term", "entrez_id")])
    sets <- lapply(split(as.character(mapped_pairs$entrez_id), mapped_pairs$term), function(ids) sort(unique(ids)))
    term_audit <- data.frame(
        term = names(gene_sets),
        input_symbols = lengths(gene_sets),
        mapped_unique_entrez = lengths(sets[names(gene_sets)]),
        stringsAsFactors = FALSE
    )
    term_audit$mapped_unique_entrez[is.na(term_audit$mapped_unique_entrez)] <- 0L
    utils::write.csv(term_audit, file.path(output_dir, "gene_set_term_mapping_audit.csv"), row.names = FALSE)
    mapped_symbols <- unique(mapped[mapped$mapped, c("symbol", "entrez_id")])
    audit <- data.frame(
        library = library_name,
        source = normalizePath(file.path(gene_set_dir, paste0(library_name, ".gmt")), mustWork = TRUE),
        mapping_method = "org.Hs.eg.db SYMBOL to all ENTREZID mappings",
        input_terms = length(gene_sets),
        input_unique_symbols = length(unique(pairs$symbol)),
        mapped_unique_symbols = length(unique(mapped_symbols$symbol)),
        unmapped_unique_symbols = length(setdiff(unique(pairs$symbol), mapped_symbols$symbol)),
        unique_symbol_entrez_pairs = nrow(mapped_symbols),
        mapped_unique_entrez_ids = length(unique(mapped_symbols$entrez_id)),
        terms_with_any_entrez_id = sum(lengths(sets) > 0L),
        terms_with_at_least_5_entrez_ids = sum(lengths(sets) >= min_gene_set_size),
        stringsAsFactors = FALSE
    )
    utils::write.csv(audit, file.path(output_dir, "gene_set_mapping_audit.csv"), row.names = FALSE)
    sets
}

model_output_dir <- file.path(output_root, model, library_name)
if (dir.exists(model_output_dir) && length(list.files(model_output_dir, all.files = TRUE, no.. = TRUE))) {
    stop("Refusing to overwrite non-empty result directory: ", model_output_dir)
}
dir.create(model_output_dir, recursive = TRUE, showWarnings = FALSE)

gene_sets <- if (library_name == "go_bp") {
    utils::write.csv(
        data.frame(term = character(), symbol = character(), entrez_id = character(), mapped = logical()),
        file.path(model_output_dir, "gene_set_symbol_entrez_mapping.csv"), row.names = FALSE
    )
    utils::write.csv(
        data.frame(
            library = library_name,
            source = "rGREAT built-in GO:BP (org.Hs.eg.db annotations; GO.db ontology metadata)",
            mapping_method = "built-in Entrez-ID GO gene sets; no SYMBOL conversion",
            stringsAsFactors = FALSE
        ),
        file.path(model_output_dir, "gene_set_mapping_audit.csv"), row.names = FALSE
    )
    "GO:BP"
} else {
    gmt_path <- file.path(gene_set_dir, paste0(library_name, ".gmt"))
    map_symbol_sets_to_entrez(rGREAT::read_gmt(gmt_path), model_output_dir)
}

background_table <- read_bed(background_path)
if (nrow(background_table) != background_regions_expected) {
    stop("Expected 896,980 union-background CpGs")
}
background <- to_granges(background_table)
background_lookup <- setNames(
    paste(background_table$chromosome, background_table$start, background_table$end, sep = ":"),
    background_table$cpg
)
cores <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")))
if (is.na(cores) || cores < 1L) cores <- 1L

for (direction in c("top", "bottom")) {
    foreground_path <- file.path(input_dir, model, paste0(direction, "_foreground.bed"))
    foreground_table <- read_bed(foreground_path)
    if (nrow(foreground_table) != foreground_regions_expected) stop("Expected 1,000 foreground CpGs")
    observed_keys <- paste(
        foreground_table$chromosome, foreground_table$start, foreground_table$end, sep = ":"
    )
    expected_keys <- unname(background_lookup[foreground_table$cpg])
    if (anyNA(expected_keys) || !identical(observed_keys, expected_keys)) {
        stop("Foreground is not an exact subset of the union background")
    }
    foreground <- to_granges(foreground_table)
    result <- rGREAT::great(
        foreground,
        gene_sets = gene_sets,
        tss_source = "GREAT:hg19",
        min_gene_set_size = min_gene_set_size,
        mode = config$mode,
        extend_from = config$extend_from,
        basal_upstream = as.integer(config$basal_upstream_bp),
        basal_downstream = as.integer(config$basal_downstream_bp),
        extension = as.integer(config$extension_bp),
        background = background,
        exclude = NULL,
        cores = cores,
        verbose = TRUE
    )
    enrichment <- as.data.frame(rGREAT::getEnrichmentTable(result, min_region_hits = min_region_hits))
    enrichment <- enrichment[order(enrichment$p_adjust, enrichment$p_value), , drop = FALSE]
    output_dir <- file.path(output_root, model, library_name, direction)
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(enrichment, file.path(output_dir, paste0(library_name, "_all.csv")), row.names = FALSE)
    utils::write.csv(
        enrichment[enrichment$p_adjust < 0.05, , drop = FALSE],
        file.path(output_dir, paste0(library_name, "_fdr05.csv")),
        row.names = FALSE
    )
    association_ranges <- rGREAT::getRegionGeneAssociations(result, use_symbols = TRUE)
    associations <- as.data.frame(association_ranges)
    association_keys <- paste(
        as.character(seqnames(association_ranges)), start(association_ranges) - 1L,
        end(association_ranges), sep = ":"
    )
    matched_foreground <- match(association_keys, observed_keys)
    if (anyNA(matched_foreground)) stop("Could not match GREAT associations to foreground CpGs")
    associations <- data.frame(cpg = foreground_table$cpg[matched_foreground], associations, check.names = FALSE)
    metadata <- S4Vectors::mcols(association_ranges)
    for (column in c("annotated_genes", "dist_to_TSS")) {
        associations[[column]] <- vapply(metadata[[column]], paste, collapse = ";", FUN.VALUE = character(1))
    }
    utils::write.csv(associations, file.path(output_dir, "region_gene_associations.csv"), row.names = FALSE)
    writeLines(
        c(
            "method\tlocal rGREAT restricted-background binomial",
            paste0("rGREAT_version\t", config$rGREAT_version),
            "genome\thg19/GRCh37",
            paste0("model\t", model),
            paste0("library\t", library_name),
            "tss_source\tGREAT:hg19",
            paste0("min_gene_set_size\t", min_gene_set_size),
            paste0("min_region_hits\t", min_region_hits),
            paste0("mode\t", config$mode),
            paste0("extend_from\t", config$extend_from),
            paste0("basal_upstream_bp\t", config$basal_upstream_bp),
            paste0("basal_downstream_bp\t", config$basal_downstream_bp),
            paste0("extension_bp\t", config$extension_bp),
            "exclude\tNULL",
            "background\t27K v1.2 + 450K v1.2 + EPIC v1.0 B4 union",
            "background_regions\t896980",
            "foreground_regions\t1000",
            "multiple_testing\tBH within model-direction-library after min_region_hits filter"
        ),
        file.path(output_dir, "parameters.tsv")
    )
    writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
}
