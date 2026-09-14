#!/usr/bin/env bash
# Nested nucleus-offset B2B for tone_ctx_v1.
#
#   MODEL=M0 bash scripts/submit_tone_ctx_v1.sh --pilot
#   MODEL=M3 bash scripts/submit_tone_ctx_v1.sh
#   MODEL=M3 bash scripts/submit_tone_ctx_v1.sh --existence
#   MODEL=M3 bash scripts/submit_tone_ctx_v1.sh --splithalf
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
MODEL="${MODEL:?set MODEL=M0|M1|M2|M3}"
# Pre-registered pilot IDs (2026-09-03). No group contrast on these alone.
PILOT_SUBJECTS="${PILOT_SUBJECTS:-RN102,RN103,RN105,D001d,D011d}"
MODE="observed"
PILOT=0
for arg in "$@"; do
  case "$arg" in
    --pilot) PILOT=1 ;;
    --existence) MODE="existence" ;;
    --splithalf) MODE="splithalf" ;;
    *) echo "unknown arg $arg"; exit 1 ;;
  esac
done
# shellcheck disable=SC1091
source scripts/tone_ctx_v1_env.sh
bash scripts/wire_nucleuslocked_b2b_extractors.sh

if [[ "$MODE" == "observed" ]]; then
  mkdir -p "${B2B_BASIS_DIR}"
  if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
    module load julia/1.12.6
    julia --project=.. prepare_feature_basis.jl 2>&1 | tee "logs/tone_ctx_v1_${MODEL}_prepare_basis.out"
  fi
  EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
  EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR},B2B_OUTDIR=${B2B_OUTDIR},B2B_BASIS_DIR=${B2B_BASIS_DIR}"
  EXPORT+=",B2B_PASSTHROUGH=1,B2B_H_RIDGE_KAPPA=0,B2B_PCA_K=1,B2B_PCA_FIT_MAX=1,B2B_DRIFT_KNOTS=0"
  EXPORT+=",B2B_EPOCH_ANCHOR=${B2B_EPOCH_ANCHOR},B2B_TMIN_S=${B2B_TMIN_S},B2B_TMAX_S=${B2B_TMAX_S}"
  EXPORT+=",B2B_FEATURE_SETS=${B2B_FEATURE_SETS},B2B_FAMILY_NAMES=${B2B_FAMILY_NAMES}"
  EXPORT+=",B2B_ALLOW_ZERO_FEATURE_ROWS=1,B2B_STRICT_FEAT_DIMS=0,B2B_NO_PLOT=1"
  EXPORT+=",B2B_EXPECTED_N_WORDS=${B2B_EXPECTED_N_WORDS},B2B_EXPECTED_SEC_LENGTHS=${B2B_EXPECTED_SEC_LENGTHS}"
  EXPORT+=",JULIA_CPU_TARGET=generic"
  ARRAY="1-63%16"
  if [[ "$PILOT" == "1" ]]; then
    EXPORT+=",B2B_PILOT_SUBJECTS=${PILOT_SUBJECTS}"
    ARRAY="1-5"
  fi
  JOB_ID=$(sbatch --parsable --job-name="tctx_${MODEL}" --partition="${PARTITION:-mit_normal}" \
    --exclude=node1617 --array="${ARRAY}" --export="${EXPORT}" \
    scripts/run_tone_ctx_v1_observed_array.sh)
  echo "Submitted tone_ctx_v1 ${MODEL} observed → ${JOB_ID}  out=${B2B_OUTDIR}"
  echo "${JOB_ID}" > "logs/tone_ctx_v1_${MODEL}_observed_job_id.txt"
  echo "${JOB_ID}"
  exit 0
fi

export B2B_OBSERVED_OUTDIR="${B2B_OUTDIR}"
if [[ "$MODE" == "existence" ]]; then
  export B2B_OUTDIR="${POOL}/encoding_results_b2b_tone_ctx_v1_${MODEL}_nucleus_offset_existence_nulls"
  export B2B_EXISTENCE_MIN_SHIFT="${B2B_EXISTENCE_MIN_SHIFT:-20}"
  export B2B_EXISTENCE_N_NULL="${B2B_EXISTENCE_N_NULL:-5000}"
  export B2B_N_NULL="${B2B_EXISTENCE_N_NULL}"
  test -f "${B2B_OBSERVED_OUTDIR}/_basis/feature_basis.npz" || { echo "Missing observed basis"; exit 1; }
  mkdir -p "${B2B_OUTDIR}"
  EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
  EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR},B2B_OUTDIR=${B2B_OUTDIR},B2B_OBSERVED_OUTDIR=${B2B_OBSERVED_OUTDIR}"
  EXPORT+=",B2B_BASIS_DIR=${B2B_OBSERVED_OUTDIR}/_basis"
  EXPORT+=",B2B_PASSTHROUGH=1,B2B_PCA_K=1,B2B_PCA_FIT_MAX=1,B2B_DRIFT_KNOTS=0"
  EXPORT+=",B2B_EPOCH_ANCHOR=${B2B_EPOCH_ANCHOR},B2B_TMIN_S=${B2B_TMIN_S},B2B_TMAX_S=${B2B_TMAX_S}"
  EXPORT+=",B2B_FEATURE_SETS=${B2B_FEATURE_SETS},B2B_FAMILY_NAMES=${B2B_FAMILY_NAMES}"
  EXPORT+=",B2B_ALLOW_ZERO_FEATURE_ROWS=1,B2B_STRICT_FEAT_DIMS=0,B2B_NO_PLOT=1"
  EXPORT+=",B2B_EXPECTED_N_WORDS=${B2B_EXPECTED_N_WORDS},B2B_EXPECTED_SEC_LENGTHS=${B2B_EXPECTED_SEC_LENGTHS}"
  EXPORT+=",B2B_EXISTENCE_MIN_SHIFT=${B2B_EXISTENCE_MIN_SHIFT},B2B_EXISTENCE_N_NULL=${B2B_EXISTENCE_N_NULL},B2B_N_NULL=${B2B_N_NULL}"
  EXPORT+=",B2B_EXISTENCE_NULL=freedman_lane"
  EXPORT+=",JULIA_CPU_TARGET=generic"
  JOB_ID=$(sbatch --parsable --job-name="tctxe_${MODEL}" --partition="${PARTITION:-mit_normal}" \
    --exclude=node1617 --time=8:00:00 --mem=16G --export="${EXPORT}" \
    scripts/run_existence_null_array.sh)
  echo "Submitted existence ${MODEL} → ${JOB_ID}"
  echo "${JOB_ID}" > "logs/tone_ctx_v1_${MODEL}_existence_job_id.txt"
  echo "${JOB_ID}"
  exit 0
fi

export B2B_OUTDIR="${POOL}/encoding_results_b2b_tone_ctx_v1_${MODEL}_nucleus_offset_splithalf"
test -f "${B2B_OBSERVED_OUTDIR}/_basis/feature_basis.npz" || { echo "Missing observed basis"; exit 1; }
mkdir -p "${B2B_OUTDIR}"
EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR},B2B_OUTDIR=${B2B_OUTDIR},B2B_OBSERVED_OUTDIR=${B2B_OBSERVED_OUTDIR}"
EXPORT+=",B2B_BASIS_DIR=${B2B_OBSERVED_OUTDIR}/_basis"
EXPORT+=",B2B_PASSTHROUGH=1,B2B_PCA_K=1,B2B_PCA_FIT_MAX=1,B2B_DRIFT_KNOTS=0"
EXPORT+=",B2B_EPOCH_ANCHOR=${B2B_EPOCH_ANCHOR},B2B_TMIN_S=${B2B_TMIN_S},B2B_TMAX_S=${B2B_TMAX_S}"
EXPORT+=",B2B_FEATURE_SETS=${B2B_FEATURE_SETS},B2B_FAMILY_NAMES=${B2B_FAMILY_NAMES}"
EXPORT+=",B2B_ALLOW_ZERO_FEATURE_ROWS=1,B2B_STRICT_FEAT_DIMS=0,B2B_NO_PLOT=1"
EXPORT+=",B2B_EXPECTED_N_WORDS=${B2B_EXPECTED_N_WORDS},B2B_EXPECTED_SEC_LENGTHS=${B2B_EXPECTED_SEC_LENGTHS}"
EXPORT+=",JULIA_CPU_TARGET=generic"
JOB_ID=$(sbatch --parsable --job-name="tctxs_${MODEL}" --partition="${PARTITION:-mit_normal}" \
  --exclude=node1617 --time=4:00:00 --export="${EXPORT}" scripts/run_split_half_array.sh)
echo "Submitted split-half ${MODEL} → ${JOB_ID}"
echo "${JOB_ID}" > "logs/tone_ctx_v1_${MODEL}_splithalf_job_id.txt"
echo "${JOB_ID}"
