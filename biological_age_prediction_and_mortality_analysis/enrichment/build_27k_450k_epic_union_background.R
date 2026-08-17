#!/usr/bin/env Rscript

# Build and audit the hg19 CpG universe from maintained Bioconductor annotations.
suppressPackageStartupMessages({
    library(data.table)
    library(digest)
    library(GenomicRanges)
    library(GenomeInfoDb)
    library(jsonlite)
    library(minfi)
    library(rtracklayer)
})

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1L]]), mustWork = TRUE)
manifest_path <- file.path(dirname(dirname(script_path)), "analysis_manifest.json")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = TRUE)
config <- manifest$enrichment

default_output <- "results/enrichment/array_union"
default_chain <- "/path/to/your/file" # fill in your path
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) args[[1L]] else default_output
chain_path <- if (length(args) >= 2L) args[[2L]] else default_chain
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE))) {
    stop("Refusing to overwrite a non-empty array-union directory: ", output_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

specifications <- list(
    `27K` = c("IlluminaHumanMethylation27kanno.ilmn12.hg19", "IlluminaHumanMethylation27kanno.ilmn12.hg19"),
    `450K` = c("IlluminaHumanMethylation450kanno.ilmn12.hg19", "IlluminaHumanMethylation450kanno.ilmn12.hg19"),
    EPIC = c("IlluminaHumanMethylationEPICanno.ilm10b4.hg19", "IlluminaHumanMethylationEPICanno.ilm10b4.hg19")
)
expected_versions <- c(
    IlluminaHumanMethylation27kanno.ilmn12.hg19 = "0.6.0",
    IlluminaHumanMethylation450kanno.ilmn12.hg19 = "0.6.1",
    IlluminaHumanMethylationEPICanno.ilm10b4.hg19 = "0.6.0"
)
canonical_chromosomes <- paste0("chr", c(seq_len(22L), "X", "Y"))

sha256 <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)

load_locations <- function(specification, platform) {
    package <- specification[[1L]]
    object <- specification[[2L]]
    if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
    observed <- as.character(utils::packageVersion(package))
    if (!identical(observed, unname(expected_versions[[package]]))) {
        stop("Unexpected ", package, " version: ", observed)
    }
    holder <- new.env(parent = emptyenv())
    utils::data(list = object, package = package, envir = holder)
    ranges <- minfi::getLocations(get(object, envir = holder, inherits = FALSE))
    valid <- !is.na(names(ranges)) & grepl("^cg[0-9]+$", names(ranges)) &
        as.character(seqnames(ranges)) %in% canonical_chromosomes &
        !is.na(start(ranges)) & start(ranges) > 0L & width(ranges) == 1L
    ranges <- ranges[valid]
    strand(ranges) <- "*"
    if (anyDuplicated(names(ranges))) stop(platform, " contains duplicate CpG IDs")
    ranges
}

coordinate_key <- function(ranges) paste(as.character(seqnames(ranges)), start(ranges), sep = ":")

compare_shared <- function(left, right) {
    shared <- intersect(names(left), names(right))
    same <- coordinate_key(left[match(shared, names(left))]) == coordinate_key(right[match(shared, names(right))])
    list(shared = length(shared), matches = sum(same), conflicts = shared[!same])
}

to_table <- function(ranges, platform) data.table(
    cpg = names(ranges), chromosome = as.character(seqnames(ranges)),
    position = as.integer(start(ranges)), platform = platform
)

write_bed <- function(table, path) {
    fwrite(
        table[, .(chromosome, start = position - 1L, end = position, cpg)],
        path, sep = "\t", col.names = FALSE, quote = FALSE, compress = "gzip"
    )
}

ranges_27k_source <- load_locations(specifications$`27K`, "27K")
ranges_450k <- load_locations(specifications$`450K`, "450K")
ranges_epic <- load_locations(specifications$EPIC, "EPIC v1.0 B4")
# The 27K v1.2 object reports hg19 but contains NCBI36 coordinates. The formal
# audit requires the near-zero pre-lift agreement and exact post-lift agreement.
before_27k_450k <- compare_shared(ranges_27k_source, ranges_450k)
if (before_27k_450k$shared < 25000L || before_27k_450k$matches > 100L) {
    stop("27K package coordinates do not show the expected hg18 signature")
}
if (!file.exists(chain_path)) stop("Missing hg18-to-hg19 chain: ", chain_path)
genome(ranges_27k_source) <- "hg18"
lifted <- rtracklayer::liftOver(ranges_27k_source, rtracklayer::import.chain(chain_path))
lift_counts <- S4Vectors::elementNROWS(lifted)
if (any(lift_counts > 1L)) stop("27K liftOver produced multiply mapped probes")
ranges_27k <- unlist(lifted[lift_counts == 1L], use.names = FALSE)
names(ranges_27k) <- names(lifted)[lift_counts == 1L]
genome(ranges_27k) <- "hg19"
strand(ranges_27k) <- "*"
if (length(ranges_27k) != 27574L) stop("Expected 27,574 uniquely lifted 27K probes")
after_27k_450k <- compare_shared(ranges_27k, ranges_450k)
if (after_27k_450k$matches != after_27k_450k$shared) {
    stop("Lifted 27K coordinates do not exactly agree with shared 450K probes")
}
after_27k_epic <- compare_shared(ranges_27k, ranges_epic)
after_450k_epic <- compare_shared(ranges_450k, ranges_epic)
if (length(after_27k_450k$conflicts) || length(after_27k_epic$conflicts) || length(after_450k_epic$conflicts)) {
    stop("Cross-platform CpG IDs have conflicting hg19 coordinates")
}

table_27k <- to_table(ranges_27k, "27K")
table_450k <- to_table(ranges_450k, "450K")
table_epic <- to_table(ranges_epic, "EPIC")
combined <- rbindlist(list(table_450k, table_epic, table_27k))
identifier_conflicts <- combined[, .(coordinates = uniqueN(paste(chromosome, position))), by = cpg][coordinates > 1L]
if (nrow(identifier_conflicts)) stop("A CpG ID has conflicting coordinates across arrays")
union <- unique(combined, by = "cpg")
union[, `:=`(
    in_27k = cpg %chin% table_27k$cpg,
    in_450k = cpg %chin% table_450k$cpg,
    in_epic = cpg %chin% table_epic$cpg,
    coordinate_source = fifelse(cpg %chin% table_450k$cpg, "450K_v1.2",
        fifelse(cpg %chin% table_epic$cpg, "EPIC_v1.0_B4", "27K_v1.2_liftOver"))
)]
coordinate_duplicates <- union[, .N, by = .(chromosome, position)][N > 1L]
if (nrow(coordinate_duplicates)) stop("Distinct CpG IDs share a union-background coordinate")
union[, chromosome_order := match(chromosome, canonical_chromosomes)]
setorder(union, chromosome_order, position, cpg)
union[, chromosome_order := NULL]

expected_counts <- unlist(config$platform_counts)
observed_counts <- c(in_27k = nrow(table_27k), in_450k = nrow(table_450k), in_epic = nrow(table_epic))
if (!identical(as.integer(observed_counts), as.integer(expected_counts[names(observed_counts)])) ||
    nrow(union) != as.integer(config$union_cpgs)) {
    stop("Array annotation counts differ from the analysis manifest")
}

membership_path <- file.path(output_dir, "platform_membership.csv.gz")
background_path <- file.path(output_dir, "background_27k_450k_epic_union_hg19.bed.gz")
fwrite(union, membership_path, compress = "gzip")
write_bed(union, background_path)
fwrite(
    data.table(platform = c("27K", "450K", "EPIC", "union"),
               cpg_count = c(observed_counts, nrow(union))),
    file.path(output_dir, "platform_counts.tsv"), sep = "\t"
)
audit <- list(
    analysis_manifest = list(path = manifest_path, sha256 = sha256(manifest_path)),
    genome = config$genome,
    bed_coordinate_convention = "0-based, half-open [position-1, position)",
    annotation_packages = as.list(expected_versions),
    platform_counts = as.list(c(observed_counts, union = nrow(union))),
    liftover_27k = list(
        status = "applied_hg18_to_hg19",
        chain_path = chain_path,
        chain_sha256 = sha256(chain_path),
        source_regions = length(ranges_27k_source),
        uniquely_lifted_regions = length(ranges_27k),
        unmapped_regions = sum(lift_counts == 0L),
        multiply_mapped_regions = sum(lift_counts > 1L),
        shared_450k_before = before_27k_450k$shared,
        matches_450k_before = before_27k_450k$matches,
        shared_450k_after = after_27k_450k$shared,
        matches_450k_after = after_27k_450k$matches
    ),
    cross_platform = list(
        shared_27k_epic = after_27k_epic$shared,
        shared_450k_epic = after_450k_epic$shared,
        identifier_coordinate_conflicts = nrow(identifier_conflicts),
        duplicate_union_coordinates = nrow(coordinate_duplicates)
    ),
    outputs = list(
        membership = list(path = membership_path, sha256 = sha256(membership_path)),
        background_bed = list(path = background_path, sha256 = sha256(background_path))
    )
)
jsonlite::write_json(audit, file.path(output_dir, "annotation_audit.json"), auto_unbox = TRUE, pretty = TRUE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
