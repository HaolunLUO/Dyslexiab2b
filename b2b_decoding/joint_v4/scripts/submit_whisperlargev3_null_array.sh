#!/bin/bash
# Submit stimulus-shift null array for Whisper large-v3 B2B (onset or offset via env).
#
# Usage (from joint_v4/):
#   bash scripts/submit_whisperlargev3_null_array.sh              # onset
#   B2B_EPOCH_ANCHOR=offset bash scripts/submit_whisperlargev3_null_array.sh
#
# 63 participants × 5 chunks × 40 null reps = 315 array tasks per anchor.
#
# Tuned for QOSMaxCpuPerUserLimit ≈ 96 CPUs (do not queue offset while onset
# nulls still hold the CPU budget). Override with env:
#   NULL_ARRAY_CONCUR=24 NULL_MEM=4G NULL_TIME=2:00:00

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"

export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_largev3_b2b}"
export B2B_EPOCH_ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
if [[ "$B2B_EPOCH_ANCHOR" == "offset" ]]; then
  export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_offset_joint_k8}"
else
  export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_onset_joint_k8}"
fi
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=null
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_FEAT_ACOUSTIC=whisperlargev3_acoustic
export B2B_FEAT_SPEECH=whisperlargev3_speech
export B2B_FEAT_LANGUAGE=whisperlargev3_language_audio_fused
export B2B_N_NULL=200
export B2B_NULL_COUNT=40
export B2B_NO_PLOT=1

# Match ~96 CPU QOS cap at -c 4; mem/time from measured null chunk stats.
NULL_ARRAY_CONCUR="${NULL_ARRAY_CONCUR:-24}"
NULL_MEM="${NULL_MEM:-4G}"
NULL_TIME="${NULL_TIME:-2:00:00}"
NULL_CPUS="${NULL_CPUS:-4}"

test -f "${B2B_BASIS_DIR}/feature_basis.npz" || {
  echo "Missing basis: ${B2B_BASIS_DIR}/feature_basis.npz"
  exit 1
}

TAG="v3_${B2B_EPOCH_ANCHOR}"
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
EXPORT+=",B2B_N_NULL=${B2B_N_NULL}"
EXPORT+=",B2B_NULL_COUNT=${B2B_NULL_COUNT}"
EXPORT+=",B2B_NO_PLOT=${B2B_NO_PLOT}"

# Default full array; set NULL_ARRAY_SPEC to resume missing tasks only
# (e.g. NULL_ARRAY_SPEC=$(cat logs/whisperlargev3_null_onset_missing_array.txt)).
NULL_ARRAY_SPEC="${NULL_ARRAY_SPEC:-1-315}"
ARRAY_ARG="${NULL_ARRAY_SPEC}%${NULL_ARRAY_CONCUR}"

SBATCH_EXTRA=()
if [[ -n "${NULL_DEPENDENCY:-}" ]]; then
  SBATCH_EXTRA+=(--dependency="${NULL_DEPENDENCY}")
fi

echo "=== Large-v3 null array (${B2B_EPOCH_ANCHOR}) ==="
echo "  outdir : ${B2B_OUTDIR}"
echo "  array  : ${ARRAY_ARG}  cpus=${NULL_CPUS}  mem=${NULL_MEM}  time=${NULL_TIME}"

JOB_ID=$(sbatch \
  --job-name="b2b_${TAG}_null" \
  --array="${ARRAY_ARG}" \
  --cpus-per-task="${NULL_CPUS}" \
  --mem="${NULL_MEM}" \
  --time="${NULL_TIME}" \
  --export="${EXPORT}" \
  "${SBATCH_EXTRA[@]}" \
  scripts/run_null_array.sh | awk '{print $4}')

echo "Submitted null array → job ${JOB_ID}"
echo "${JOB_ID}" > "logs/whisperlargev3_null_${B2B_EPOCH_ANCHOR}_array_job_id.txt"
