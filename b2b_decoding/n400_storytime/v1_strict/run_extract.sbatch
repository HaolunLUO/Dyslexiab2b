#!/bin/bash
#SBATCH -J n400_v1s_ext
#SBATCH -p mit_quicktest
#SBATCH --array=0-7
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH --open-mode=append
#SBATCH -o /orcd/pool/005/haolun52/n400_storytime_v1/v1_strict/models/logs/extract_%A_%a.out
#SBATCH -e /orcd/pool/005/haolun52/n400_storytime_v1/v1_strict/models/logs/extract_%A_%a.err
# Channel means and figure sums. mit_quicktest is 15 min; 8 tasks cover 63 subjects.

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run outside Slurm." >&2
  exit 1
fi
PY=/orcd/software/core/001/pkg/miniforge/25.11.0-0/bin/python3
ROOT=/orcd/pool/005/haolun52/_work_Dyslexiab2b_n400rev/b2b_decoding/n400_storytime/v1_strict
SUBS=/orcd/pool/005/haolun52/dyslexia_natualistics_listing/b2b_decoding/subjects.txt
mapfile -t ALL < "$SUBS"
n=${#ALL[@]}
start=$((SLURM_ARRAY_TASK_ID * 8))
end=$((start + 8))
if (( end > n )); then end=$n; fi
if (( start >= n )); then
  echo "task ${SLURM_ARRAY_TASK_ID} has no subjects"
  exit 0
fi
chunk=("${ALL[@]:start:end-start}")
echo "host $(hostname) job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} ${chunk[*]}"
"$PY" "$ROOT/extract_channels.py" "${chunk[@]}"
echo "end $(date -Is)"
