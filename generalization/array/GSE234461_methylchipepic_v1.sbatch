#!/bin/bash
#SBATCH --job-name=GSE234461_methylchipepic_v1
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python -m generalization.array.GSE234461_methylchipepic_v1 --size-index "${SLURM_ARRAY_TASK_ID}"
