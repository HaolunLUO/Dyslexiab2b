#!/bin/bash
# Pilot B2B v4 with Hasson-aligned Whisper large-v3 features.
#
# Usage:
#   bash scripts/run_whisperlargev3_pilot.sh
#   bash scripts/run_whisperlargev3_pilot.sh RN109 D007d
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_largev3_b2b}"
export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_onset_joint_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=observed
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR=onset
export B2B_FEAT_ACOUSTIC=whisperlargev3_acoustic
export B2B_FEAT_SPEECH=whisperlargev3_speech
export B2B_FEAT_LANGUAGE=whisperlargev3_language_audio_fused
export B2B_NO_PLOT=1
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-4}"

module load julia/1.12.6

echo "=== B2B pilot: Whisper large-v3 (Hasson-aligned) ==="
echo "  extractor: $B2B_EXTRACTOR_DIR"
echo "  outdir   : $B2B_OUTDIR"
echo "  features : $B2B_FEAT_ACOUSTIC / $B2B_FEAT_SPEECH / $B2B_FEAT_LANGUAGE"

test -d "$B2B_EXTRACTOR_DIR/_shared_wordlocked_features/section_001" || {
  echo "Missing overlay; run: bash scripts/wire_whisperlargev3_b2b_extractors.sh"
  exit 1
}

if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "Preparing large-v3 feature basis…"
  julia --project=.. prepare_feature_basis.jl 2>&1 | tee "logs/whisperlargev3_prepare_basis.out"
fi

SUBJECTS=("$@")
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  SUBJECTS=(RN109 D007d D011d)
fi

for p in "${SUBJECTS[@]}"; do
  echo "======== LARGE-V3 PILOT $p ========"
  julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/whisperlargev3_pilot_${p}.out"
done

echo "Large-v3 pilot complete → $B2B_OUTDIR"
