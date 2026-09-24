#!/bin/bash
#SBATCH -J n400_rev_beh
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/n400_rev_beh_%j.out
#SBATCH -e logs/n400_rev_beh_%j.err
# Phase 4. The R script exits 0 without fitting when the split-half gate is closed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
source /etc/profile.d/00-modulepath.sh
module load miniforge/25.11.0-0
module load R/4.5.2
export N400_REV_DIR="${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
Rscript fit_behaviour.R
echo "end $(date -Is)"
