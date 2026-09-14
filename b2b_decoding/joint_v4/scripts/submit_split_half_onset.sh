#!/bin/bash
# Phase 2 split-half — onset (primary).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
export B2B_EXTRACTOR_DIR="${POOL}/extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b"
export B2B_EPOCH_ANCHOR=onset
export B2B_TMIN_S=-0.3
export B2B_TMAX_S=1.0
export B2B_OBSERVED_OUTDIR="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_tonev3_onset_tmin03_passthrough"
export B2B_OUTDIR="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_tonev3_onset_tmin03_splithalf"
export B2B_BASIS_DIR="${B2B_OBSERVED_OUTDIR}/_basis"
export B2B_PASSTHROUGH=1
export B2B_H_RIDGE_KAPPA=0
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_STRICT_FEAT_DIMS=0
export B2B_NO_PLOT=1
export B2B_FEATURE_SETS=envelope_v2:pitch:tone_v3:offset:lexical_frequency:gpt2cn_surprisal
export B2B_FAMILY_NAMES=envelope:pitch:tone:offset:frequency:surprisal
JOB_NAME="${JOB_NAME:-b2b_sh_on}"
PARTITION="${PARTITION:-mit_normal}"

test -f "${B2B_BASIS_DIR}/feature_basis.npz" || { echo "Missing basis"; exit 1; }
mkdir -p "${B2B_OUTDIR}"

EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR}"
EXPORT+=",B2B_OUTDIR=${B2B_OUTDIR}"
EXPORT+=",B2B_OBSERVED_OUTDIR=${B2B_OBSERVED_OUTDIR}"
EXPORT+=",B2B_BASIS_DIR=${B2B_BASIS_DIR}"
EXPORT+=",B2B_PASSTHROUGH=1,B2B_H_RIDGE_KAPPA=0,B2B_PCA_K=1,B2B_PCA_FIT_MAX=1,B2B_DRIFT_KNOTS=0"
EXPORT+=",B2B_EPOCH_ANCHOR=${B2B_EPOCH_ANCHOR},B2B_TMIN_S=${B2B_TMIN_S},B2B_TMAX_S=${B2B_TMAX_S}"
EXPORT+=",B2B_FEATURE_SETS=${B2B_FEATURE_SETS},B2B_FAMILY_NAMES=${B2B_FAMILY_NAMES}"
EXPORT+=",B2B_ALLOW_ZERO_FEATURE_ROWS=1,B2B_STRICT_FEAT_DIMS=0,B2B_NO_PLOT=1"
EXPORT+=",JULIA_CPU_TARGET=generic"

echo "=== Phase 2 split-half (onset) → ${B2B_OUTDIR} ==="
JOB_ID=$(sbatch --parsable --job-name="${JOB_NAME}" --partition="${PARTITION}" \
  --exclude=node1617 --export="${EXPORT}" scripts/run_split_half_array.sh)
echo "Submitted onset split-half → job ${JOB_ID}"
echo "${JOB_ID}" > logs/split_half_onset_job_id.txt
echo "${JOB_ID}"
