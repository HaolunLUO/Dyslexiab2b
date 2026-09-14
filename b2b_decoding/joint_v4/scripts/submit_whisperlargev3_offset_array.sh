#!/bin/bash
# Submit full-cohort B2B v4 with Whisper large-v3, offset-locked epochs.
#
# Usage (from joint_v4/):
#   mkdir -p logs
#   bash scripts/submit_whisperlargev3_offset_array.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"

export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_largev3_b2b}"
export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_offset_joint_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=observed
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR=offset
export B2B_FEAT_ACOUSTIC=whisperlargev3_acoustic
export B2B_FEAT_SPEECH=whisperlargev3_speech
export B2B_FEAT_LANGUAGE=whisperlargev3_language_audio_fused
export B2B_NO_PLOT=1

echo "=== Large-v3 B2B offset scale-up ==="
echo "  extractor: ${B2B_EXTRACTOR_DIR}"
echo "  outdir   : ${B2B_OUTDIR}"
echo "  anchor   : ${B2B_EPOCH_ANCHOR}"

bash scripts/wire_whisperlargev3_b2b_extractors.sh

module load julia/1.12.6
if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "Preparing large-v3 feature basis (offset outdir)…"
  julia --project=.. prepare_feature_basis.jl 2>&1 | tee "logs/whisperlargev3_offset_prepare_basis.out"
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

JOB_ID=$(sbatch \
  --job-name=b2b_v3_off \
  --export="${EXPORT}" \
  scripts/run_observed_array.sh | awk '{print $4}')

echo "Submitted large-v3 offset observed array → job ${JOB_ID}"
echo "${JOB_ID}" > logs/whisperlargev3_offset_array_job_id.txt
