#!/bin/bash
# Submit full-cohort B2B v4 on MNE-ICA v1 EEG with Whisper large-v3 features.
#
# Usage (from joint_v4/):
#   mkdir -p logs
#   bash scripts/submit_mne_ica_v1_largev3_array.sh              # onset
#   B2B_EPOCH_ANCHOR=offset bash scripts/submit_mne_ica_v1_largev3_array.sh
#
# Reuses the frozen large-v3 feature basis (stimulus-only). EEG comes from ICA v1.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"

export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_mne_ica_v1_largev3_b2b}"
export B2B_EPOCH_ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
if [[ "$B2B_EPOCH_ANCHOR" == "offset" ]]; then
  export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_offset_joint_k8_mne_ica_v1}"
  SRC_BASIS="${SRC_BASIS:-${POOL}/encoding_results_b2b_largev3_offset_joint_k8/_basis}"
  JOB_NAME="${JOB_NAME:-b2b_ica_off}"
else
  export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_onset_joint_k8_mne_ica_v1}"
  SRC_BASIS="${SRC_BASIS:-${POOL}/encoding_results_b2b_largev3_onset_joint_k8/_basis}"
  JOB_NAME="${JOB_NAME:-b2b_ica_on}"
fi
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=observed
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_FEAT_ACOUSTIC=whisperlargev3_acoustic
export B2B_FEAT_SPEECH=whisperlargev3_speech
export B2B_FEAT_LANGUAGE=whisperlargev3_language_audio_fused
export B2B_NO_PLOT=1

echo "=== MNE-ICA v1 + large-v3 B2B (${B2B_EPOCH_ANCHOR}) ==="
echo "  extractor: ${B2B_EXTRACTOR_DIR}"
echo "  outdir   : ${B2B_OUTDIR}"

bash scripts/wire_whisperlargev3_b2b_extractors.sh
bash scripts/wire_mne_ica_v1_largev3_b2b_extractors.sh

mkdir -p "${B2B_BASIS_DIR}"
if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  if [[ -f "${SRC_BASIS}/feature_basis.npz" ]]; then
    echo "Copying frozen large-v3 basis from ${SRC_BASIS}"
    cp -a "${SRC_BASIS}/." "${B2B_BASIS_DIR}/"
  else
    echo "Preparing large-v3 feature basis…"
    module load julia/1.12.6
    julia --project=.. prepare_feature_basis.jl 2>&1 | tee "logs/mne_ica_v1_largev3_${B2B_EPOCH_ANCHOR}_prepare_basis.out"
  fi
else
  echo "Basis already present: ${B2B_BASIS_DIR}/feature_basis.npz"
fi

EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR}"
EXPORT+=",B2B_OUTDIR=${B2B_OUTDIR}"
EXPORT+=",B2B_BASIS_DIR=${B2B_BASIS_DIR}"
EXPORT+=",B2B_MODE=${B2B_MODE}"
EXPORT+=",B2B_PCA_K=${B2B_PCA_K}"
EXPORT+=",B2B_BASIS_MODE=${B2B_BASIS_MODE}"
EXPORT+=",B2B_EPOCH_ANCHOR=${B2B_EPOCH_ANCHOR}"
EXPORT+=",B2B_FEAT_ACOUSTIC=${B2B_FEAT_ACOUSTIC}"
EXPORT+=",B2B_FEAT_SPEECH=${B2B_FEAT_SPEECH}"
EXPORT+=",B2B_FEAT_LANGUAGE=${B2B_FEAT_LANGUAGE}"
EXPORT+=",B2B_NO_PLOT=${B2B_NO_PLOT}"

SBATCH_EXTRA=()
if [[ -n "${OBS_DEPENDENCY:-}" ]]; then
  SBATCH_EXTRA+=(--dependency="${OBS_DEPENDENCY}")
fi

JOB_ID=$(sbatch \
  --job-name="${JOB_NAME}" \
  --export="${EXPORT}" \
  "${SBATCH_EXTRA[@]}" \
  scripts/run_observed_array.sh | awk '{print $4}')

echo "Submitted MNE-ICA v1 ${B2B_EPOCH_ANCHOR} observed array → job ${JOB_ID}"
echo "${JOB_ID}" > "logs/mne_ica_v1_largev3_${B2B_EPOCH_ANCHOR}_array_job_id.txt"
echo "${JOB_ID}"
