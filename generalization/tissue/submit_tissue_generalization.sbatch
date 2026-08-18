#!/bin/bash
#SBATCH --job-name=tissue
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPOSITORY_ROOT}"

# One Slurm CPU task owns one log-spaced sample size and all 50 resamples.
# ElasticNetCV sees the four allocated CPUs through n_jobs=-1.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python -m generalization.tissue.train_tissue_generalization
