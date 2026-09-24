#!/bin/bash
#SBATCH -J n400_rev_l2
#SBATCH -p mit_quicktest
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o logs/n400_rev_l2_%j.out
#SBATCH -e logs/n400_rev_l2_%j.err
# OLS on per-child Unfold window scores. Not a trial-level LMM.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
source /etc/profile.d/00-modulepath.sh
module load R/4.5.2
echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
Rscript second_level_unfold.R
echo "end $(date -Is)"
