#!/bin/bash
#SBATCH --job-name=array
#SBATCH --array=0-59
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPOSITORY_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
DATASETS=(GSE184221 GSE234461)
GSE_INDEX=$((SLURM_ARRAY_TASK_ID / 30))
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID % 30))
python -m generalization.array.train_array_generalization \
  --gse "${DATASETS[${GSE_INDEX}]}" --size-index "${SIZE_INDEX}"
