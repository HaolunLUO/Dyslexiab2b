#!/bin/bash
# Prepare features/basis and submit ICA v1 envelope/pitch/onset/freq/surprisal B2B.
#
# Usage (from joint_v4/):
#   mkdir -p logs
#   bash scripts/submit_mne_ica_v1_envpitch_array.sh
#   B2B_EPOCH_ANCHOR=offset bash scripts/submit_mne_ica_v1_envpitch_array.sh
#
# Optional: PARTITION=mit_preemptable  (use while mit_normal is at CPU cap)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
PY="${PY:-$(pwd)/.venv_gpt2/bin/python}"

export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b}"
export B2B_EPOCH_ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
if [[ "$B2B_EPOCH_ANCHOR" == "offset" ]]; then
  export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_offset_freq_surp_passthrough}"
  JOB_NAME="${JOB_NAME:-b2b_ica_envf}"
  SRC_BASIS="${SRC_BASIS:-${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_onset_freq_surp_passthrough/_basis}"
else
  export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_onset_freq_surp_passthrough}"
  JOB_NAME="${JOB_NAME:-b2b_ica_env}"
  SRC_BASIS=""
fi
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=observed
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
export B2B_FEATURE_SETS=envelope_v2:pitch:word_onset:lexical_frequency:gpt2cn_surprisal
export B2B_FAMILY_NAMES=envelope:pitch:onset:frequency:surprisal
PARTITION="${PARTITION:-mit_normal}"

SHARED="${POOL}/extracted_sections_wordlocked_shared/_shared_wordlocked_features"
if [[ ! -f "${SHARED}/section_001/X_word_pitch.npy" ]]; then
  echo "Extracting pitch + word-onset features…"
  "$PY" scripts/extract_pitch_onset_wordlocked.py
fi
if [[ ! -f "${SHARED}/section_001/X_word_lexical_frequency.npy" ]]; then
  module load julia/1.12.6
  B2B_EXTRACTOR_DIR="${POOL}/extracted_sections_wordlocked_shared" \
    julia --project=.. scripts/prepare_lexical_features.jl
fi

bash scripts/wire_mne_ica_v1_envpitch_b2b_extractors.sh

mkdir -p "${B2B_BASIS_DIR}"
if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  if [[ -n "${SRC_BASIS}" && -f "${SRC_BASIS}/feature_basis.npz" ]]; then
    echo "Copying frozen passthrough basis from ${SRC_BASIS}"
    cp -a "${SRC_BASIS}/." "${B2B_BASIS_DIR}/"
  else
    echo "Preparing passthrough feature basis…"
    module load julia/1.12.6
    julia --project=.. prepare_feature_basis.jl 2>&1 | tee "logs/mne_ica_v1_envpitch_${B2B_EPOCH_ANCHOR}_prepare_basis.out"
  fi
else
  echo "Basis already present: ${B2B_BASIS_DIR}/feature_basis.npz"
fi

EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR}"
EXPORT+=",B2B_OUTDIR=${B2B_OUTDIR}"
EXPORT+=",B2B_BASIS_DIR=${B2B_BASIS_DIR}"
EXPORT+=",B2B_MODE=${B2B_MODE}"
EXPORT+=",B2B_PASSTHROUGH=1"
EXPORT+=",B2B_H_RIDGE_KAPPA=0"
EXPORT+=",B2B_PCA_K=1"
EXPORT+=",B2B_PCA_FIT_MAX=1"
EXPORT+=",B2B_DRIFT_KNOTS=0"
EXPORT+=",B2B_EPOCH_ANCHOR=${B2B_EPOCH_ANCHOR}"
EXPORT+=",B2B_FEATURE_SETS=${B2B_FEATURE_SETS}"
EXPORT+=",B2B_FAMILY_NAMES=${B2B_FAMILY_NAMES}"
EXPORT+=",B2B_ALLOW_ZERO_FEATURE_ROWS=1"
EXPORT+=",B2B_STRICT_FEAT_DIMS=0"
EXPORT+=",B2B_NO_PLOT=1"

echo "=== ICA env/pitch B2B (${B2B_EPOCH_ANCHOR}) ==="
echo "  extractor: ${B2B_EXTRACTOR_DIR}"
echo "  outdir   : ${B2B_OUTDIR}"
echo "  partition: ${PARTITION}"

JOB_ID=$(sbatch --parsable --job-name="${JOB_NAME}" --partition="${PARTITION}" \
  --export="${EXPORT}" \
  scripts/run_mne_ica_v1_envpitch_observed_array.sh)
echo "Submitted ICA env/pitch ${B2B_EPOCH_ANCHOR} observed array → job ${JOB_ID}"
echo "${JOB_ID}" > "logs/mne_ica_v1_envpitch_${B2B_EPOCH_ANCHOR}_array_job_id.txt"
if [[ "$B2B_EPOCH_ANCHOR" == "onset" ]]; then
  echo "${JOB_ID}" > logs/mne_ica_v1_envpitch_array_job_id.txt
fi
echo "${JOB_ID}"
