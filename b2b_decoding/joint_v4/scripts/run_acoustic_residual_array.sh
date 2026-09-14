#!/bin/bash
#SBATCH -J hdc_acou_res
#SBATCH -p mit_normal
#SBATCH -a 1-63%16
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -o logs/hdc_acou_res_%A_%a.out
#SBATCH -e logs/hdc_acou_res_%A_%a.err
#
# Envelope + pitch TRF residualization (MNE), one primary cohort subject per task.
# Submit from joint_v4: mkdir -p logs && sbatch scripts/run_acoustic_residual_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

SRC="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
OUT="${B2B_ACOUSTIC_RESIDUAL_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
PY="${HDC_PYTHON:-}"
if [[ -z "$PY" && -x .venv_gpt2/bin/python3 ]]; then
  PY=.venv_gpt2/bin/python3
fi
if [[ -z "$PY" ]]; then
  PY=python3
fi

echo "Acoustic residual task ${SLURM_ARRAY_TASK_ID}  src=${SRC}  out=${OUT}"
"$PY" scripts/regress_out_acoustic_mne.py \
  --src "$SRC" \
  --out "$OUT" \
  --cohort-csv cohort_groups.csv \
  --cohort-index "${SLURM_ARRAY_TASK_ID}" \
  ${B2B_FORCE:+--force}
