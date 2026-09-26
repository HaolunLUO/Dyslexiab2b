#!/bin/bash
#SBATCH -J n400_v1s_plot
#SBATCH -p mit_quicktest
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH --open-mode=append
#SBATCH -o /orcd/pool/005/haolun52/n400_storytime_v1/v1_strict/models/logs/plot_%j.out
#SBATCH -e /orcd/pool/005/haolun52/n400_storytime_v1/v1_strict/models/logs/plot_%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run outside Slurm." >&2
  exit 1
fi
PY=/orcd/software/core/001/pkg/miniforge/25.11.0-0/bin/python3
ROOT=/orcd/pool/005/haolun52/_work_Dyslexiab2b_n400rev/b2b_decoding/n400_storytime/v1_strict
echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
"$PY" "$ROOT/plot_v1_strict.py"
echo "end $(date -Is)"
