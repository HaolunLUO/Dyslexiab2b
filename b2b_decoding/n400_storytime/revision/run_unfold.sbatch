#!/bin/bash
#SBATCH -J n400_unfold
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -a 1-63%12
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/n400_unfold_%A_%a.out
#SBATCH -e logs/n400_unfold_%A_%a.err
# One child per task. Does not write into epoch outdirs or the B2B Julia project.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
source /etc/profile.d/00-modulepath.sh
module load julia/1.12.6
export JULIA_DEPOT_PATH="/orcd/pool/005/haolun52/.julia"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
POOL="/orcd/pool/005/haolun52"
ENV="${POOL}/n400_storytime_v1/unfold/julia_env"
SUBJECTS="${POOL}/dyslexia_natualistics_listing/b2b_decoding/subjects.txt"
SUBJ="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SUBJECTS}")"
echo "task ${SLURM_ARRAY_TASK_ID} subject ${SUBJ} host $(hostname)"
if [[ -z "${SUBJ}" ]]; then
  echo "no subject" >&2
  exit 1
fi
julia --project="${ENV}" fit_unfold.jl "${SUBJ}"
