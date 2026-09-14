#!/bin/bash
#SBATCH -J b2b_mfa_v2_null
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -a 1-315%24
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 4:00:00
#SBATCH -o logs/b2b_mfa_v2_null_%A_%a.out
#SBATCH -e logs/b2b_mfa_v2_null_%A_%a.err
#
# Stimulus-shift nulls for MFA v2 arm.
# 63 participants × 5 chunks of 40 reps = 315 array tasks.
#
#   sbatch scripts/run_mfa_phone_tone_env_null_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=null
export B2B_EPOCH_ANCHOR=onset
export B2B_BASIS_MODE=independent
export B2B_PASSTHROUGH=1
export B2B_H_RIDGE_KAPPA=0
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_STRICT_FEAT_DIMS=0
export B2B_NO_PLOT=1
export B2B_FEATURE_SETS=envelope_v2,mfa_phones_v2,mfa_tone_v2
export B2B_FAMILY_NAMES=envelope,phones,tone
export B2B_N_NULL=200
export B2B_NULL_COUNT=40
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_mfa_v2_phone_tone_env_onset_passthrough}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"

TID="${SLURM_ARRAY_TASK_ID:?}"
PART_IDX=$(( (TID - 1) % 63 + 1 ))
CHUNK=$(( (TID - 1) / 63 ))
export B2B_NULL_START=$(( CHUNK * 40 + 1 ))
export PART_IDX

PART=$(python3 - <<'PY'
import csv, os
cohort = os.path.join(os.environ.get("SLURM_SUBMIT_DIR", "."), "cohort_groups.csv")
idx = int(os.environ["PART_IDX"])
rows = [r for r in csv.DictReader(open(cohort)) if r["include_primary"] in ("1", "True", "true")]
print(rows[idx - 1]["participant"])
PY
)

CHUNK_END=$(( B2B_NULL_START + B2B_NULL_COUNT - 1 ))
CHUNK_TAG=$(printf "%04d_%04d" "$B2B_NULL_START" "$CHUNK_END")
CHUNK_CSV="${B2B_OUTDIR}/${PART}/${PART}_null_${CHUNK_TAG}_traces.csv"
AGG_CSV="${B2B_OUTDIR}/${PART}/${PART}_null_traces.csv"
if [[ -f "$AGG_CSV" && "${B2B_FORCE:-0}" != "1" ]]; then
  echo "Skip ${PART}: aggregated null_traces already exist"
  exit 0
fi
if [[ -f "$CHUNK_CSV" && "${B2B_FORCE:-0}" != "1" ]]; then
  echo "Skip ${PART} chunk ${CHUNK_TAG}: ${CHUNK_CSV} exists"
  exit 0
fi

module load julia/1.12.6

echo "MFA v2 null task $TID  ${PART} (idx=$PART_IDX)  null_start=$B2B_NULL_START count=$B2B_NULL_COUNT  out=${B2B_OUTDIR}"
julia --project=.. b2b_joint_v4_pipeline.jl "$PART_IDX"
