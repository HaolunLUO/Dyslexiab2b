#!/bin/bash
#SBATCH -J n400_rev_audit
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/n400_rev_audit_%j.out
#SBATCH -e logs/n400_rev_audit_%j.err
# Phase 0. Loads wiki.zh.vec and as-stored EEG. Login nodes must not run this.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
source /etc/profile.d/00-modulepath.sh
module load miniforge/25.11.0-0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
python3 audit_phase0.py all
echo "end $(date -Is)"
