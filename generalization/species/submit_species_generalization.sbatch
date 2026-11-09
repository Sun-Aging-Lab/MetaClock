#!/bin/bash
#SBATCH --job-name=species
#SBATCH --array=0-299
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPOSITORY_ROOT}"

# One Slurm CPU task owns one species/sample-size point and all 50 resamples.
# GridSearchCV uses the allocated CPUs for complete pipeline fits.
SPECIES=(BosTaurus CanisLupusFamiliaris EquusCaballus HeterocephalusGlaber HomoSapiens MacacaMulatta MusMusculus OvisAries PapioHamadryas RattusNorvegicus)
export TARGET_SPECIES="${SPECIES[$((SLURM_ARRAY_TASK_ID / 30))]}"
export METACLOCK_SIZE_INDEX="$((SLURM_ARRAY_TASK_ID % 30))"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python -m generalization.species.train_species_generalization
