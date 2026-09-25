#!/bin/bash
#SBATCH -J n400_epoch
#SBATCH -p mit_quicktest
#SBATCH -a 1-63%20
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o logs/n400_epoch_%A_%a.out
#SBATCH -e logs/n400_epoch_%A_%a.err
# Epoch as-stored EEG for one subject (SLURM_ARRAY_TASK_ID -> subjects.txt).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
source /etc/profile.d/00-modulepath.sh
module load miniforge/25.11.0-0

SUBJECTS="${SUBJECTS:-/orcd/pool/005/haolun52/dyslexia_natualistics_listing/b2b_decoding/subjects.txt}"
SUBJ="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SUBJECTS}")"
if [[ -z "${SUBJ}" ]]; then
  echo "no subject for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
echo "task ${SLURM_ARRAY_TASK_ID} subject ${SUBJ}"
python3 epoch_subject.py --subject "${SUBJ}"
