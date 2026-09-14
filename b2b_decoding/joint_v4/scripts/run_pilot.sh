#!/bin/bash
# Pilot: RN109 (TD, high QC), D007d (atypical CAP), D011d (normal CAP, low retention)
# Usage: bash scripts/run_pilot.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_joint_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
export B2B_MODE=observed
export B2B_PCA_K=8
export B2B_BASIS_MODE="${B2B_BASIS_MODE:-independent}"
export B2B_EPOCH_ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
export B2B_FEAT_ACOUSTIC="${B2B_FEAT_ACOUSTIC:-whisperlargev2_acoustic}"
export B2B_FEAT_SPEECH="${B2B_FEAT_SPEECH:-whisperlargev2_speech}"
export B2B_FEAT_LANGUAGE="${B2B_FEAT_LANGUAGE:-whisperlargev2_language_audio_fused}"
export B2B_NO_PLOT=1
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-4}"

module load julia/1.12.6

if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "Preparing frozen feature basis…"
  julia --project=.. prepare_feature_basis.jl
fi

for p in RN109 D007d D011d; do
  echo "======== PILOT $p ========"
  julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/pilot_${p}.out"
done

echo "Pilot complete. Gates: ≥16 valid partitions (prefer 20), full-rank H,"
echo "  κ < 85 (investigate >32), no repeated λ-boundary hits, centered nulls later."
