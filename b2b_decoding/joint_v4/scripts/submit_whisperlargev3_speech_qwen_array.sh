#!/bin/bash
# Submit B2B: Whisper large-v3 speech + Qwen l24 (2-family), onset-locked.
#
# Usage (from joint_v4/):
#   bash scripts/submit_whisperlargev3_speech_qwen_array.sh
#   B2B_EPOCH_ANCHOR=offset bash scripts/submit_whisperlargev3_speech_qwen_array.sh
#   QWEN_LAYER=36 bash scripts/submit_whisperlargev3_speech_qwen_array.sh
#
# Pattern mirrors Tiny-speech + GPT2-CN l24.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"

export B2B_EPOCH_ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
QWEN_LAYER="${QWEN_LAYER:-24}"
QWEN_FEAT=$(printf "qwen_l%02d" "$QWEN_LAYER")

export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_largev3_b2b}"
export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3speech_${QWEN_FEAT}_${B2B_EPOCH_ANCHOR}_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=observed
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_FEATURE_SETS="whisperlargev3_speech:${QWEN_FEAT}"
export B2B_FAMILY_NAMES=speech:qwen
# Qwen/GPT2 embeddings can have all-zero rows (e.g. leading OOV/pad tokens).
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_NO_PLOT=1
# Align FEAT_* fingerprint fields with the 2-family arm (manifest lock).
export B2B_FEAT_ACOUSTIC=whisperlargev3_speech
export B2B_FEAT_SPEECH=whisperlargev3_speech
export B2B_FEAT_LANGUAGE="${QWEN_FEAT}"

# Observed is ~2 min/subj; keep modest concurrency (CPU QOS ≈ 96 → %24 at -c 4).
OBS_ARRAY_CONCUR="${OBS_ARRAY_CONCUR:-20}"
OBS_MEM="${OBS_MEM:-8G}"
OBS_TIME="${OBS_TIME:-1:00:00}"
OBS_CPUS="${OBS_CPUS:-4}"

echo "=== Large-v3 speech + ${QWEN_FEAT} (${B2B_EPOCH_ANCHOR}) ==="
echo "  extractor: ${B2B_EXTRACTOR_DIR}"
echo "  outdir   : ${B2B_OUTDIR}"
echo "  features : ${B2B_FEATURE_SETS}"

bash scripts/wire_whisperlargev3_b2b_extractors.sh

test -f "${B2B_EXTRACTOR_DIR}/_shared_wordlocked_features/section_001/X_word_whisperlargev3_speech.npy"
test -f "${B2B_EXTRACTOR_DIR}/_shared_wordlocked_features/section_001/X_word_${QWEN_FEAT}.npy"

module load julia/1.12.6
if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "Preparing feature basis…"
  julia --project=.. prepare_feature_basis.jl 2>&1 | tee "logs/largev3speech_${QWEN_FEAT}_${B2B_EPOCH_ANCHOR}_prepare_basis.out"
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
EXPORT+=",B2B_FEATURE_SETS=${B2B_FEATURE_SETS}"
EXPORT+=",B2B_FAMILY_NAMES=${B2B_FAMILY_NAMES}"
EXPORT+=",B2B_FEAT_ACOUSTIC=${B2B_FEAT_ACOUSTIC}"
EXPORT+=",B2B_FEAT_SPEECH=${B2B_FEAT_SPEECH}"
EXPORT+=",B2B_FEAT_LANGUAGE=${B2B_FEAT_LANGUAGE}"
EXPORT+=",B2B_ALLOW_ZERO_FEATURE_ROWS=${B2B_ALLOW_ZERO_FEATURE_ROWS}"
EXPORT+=",B2B_NO_PLOT=${B2B_NO_PLOT}"

TAG="v3sp_q${QWEN_LAYER}_${B2B_EPOCH_ANCHOR}"
JOB_ID=$(sbatch \
  --job-name="b2b_${TAG}" \
  --array="1-63%${OBS_ARRAY_CONCUR}" \
  --cpus-per-task="${OBS_CPUS}" \
  --mem="${OBS_MEM}" \
  --time="${OBS_TIME}" \
  --export="${EXPORT}" \
  scripts/run_observed_array.sh | awk '{print $4}')

echo "Submitted → job ${JOB_ID}"
echo "${JOB_ID}" > "logs/largev3speech_${QWEN_FEAT}_${B2B_EPOCH_ANCHOR}_array_job_id.txt"
