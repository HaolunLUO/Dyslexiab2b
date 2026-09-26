#!/bin/bash
#SBATCH -J n400_v1s_lmm
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH --open-mode=append
#SBATCH -o /orcd/pool/005/haolun52/n400_storytime_v1/v1_strict/models/logs/lmm_%j.out
#SBATCH -e /orcd/pool/005/haolun52/n400_storytime_v1/v1_strict/models/logs/lmm_%j.err
# All v1-strict LMMs. Same optimizer and wall as job 23687441.

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run outside Slurm." >&2
  exit 1
fi
PY=/orcd/software/core/001/pkg/miniforge/25.11.0-0/bin/python3
ROOT=/orcd/pool/005/haolun52/_work_Dyslexiab2b_n400rev/b2b_decoding/n400_storytime/v1_strict
cd "$ROOT"
source /etc/profile.d/00-modulepath.sh
module load R/4.5.2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "host $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
"$PY" "$ROOT/assemble_trials.py"
Rscript "$ROOT/fit_v1_strict.R"
echo "end $(date -Is)"
