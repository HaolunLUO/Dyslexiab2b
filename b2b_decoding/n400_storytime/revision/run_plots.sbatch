#!/bin/bash
#SBATCH -J n400_rev_plots
#SBATCH -p mit_quicktest
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o logs/n400_rev_plots_%j.out
#SBATCH -e logs/n400_rev_plots_%j.err
# Reads saved tables. Does not refit.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
source /etc/profile.d/00-modulepath.sh
module load miniforge/25.11.0-0
echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
python3 plot_revision.py
echo "end $(date -Is)"
