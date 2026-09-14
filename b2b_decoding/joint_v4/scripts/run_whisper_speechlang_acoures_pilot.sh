#!/bin/bash
# Whisper Large V2 speech + language on acoustic-residual EEG (envelope + pitch TRF removed).
#
# Usage:
#   bash scripts/run_whisper_speechlang_acoures_pilot.sh           # RN109 / D007d / D011d
#   bash scripts/run_whisper_speechlang_acoures_pilot.sh RN109
#
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_speechlang_acoures_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=observed
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR=onset
export B2B_FEATURE_SETS=whisperlargev2_speech,whisperlargev2_language_audio_fused
export B2B_FAMILY_NAMES=speech,language
export B2B_NO_PLOT=1
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-4}"

module load julia/1.12.6

echo "=== Whisper Large V2 speech+language (acoustic-residual EEG) ==="
echo "  extractor: $B2B_EXTRACTOR_DIR"
echo "  outdir   : $B2B_OUTDIR"
echo "  features : $B2B_FEATURE_SETS"

test -d "$B2B_EXTRACTOR_DIR" || {
  echo "Missing residual extractor; run scripts/regress_out_acoustic_mne.py first"
  exit 1
}

if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "Preparing speech+language feature basis…"
  julia --project=.. prepare_feature_basis.jl
fi

SUBJECTS=("$@")
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  SUBJECTS=(RN109 D007d D011d)
fi

for p in "${SUBJECTS[@]}"; do
  echo "======== SPEECH+LANG ACOURES $p ========"
  julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/whisper_speechlang_acoures_${p}.out"
done

echo "Whisper speech+language acoustic-residual pilot complete → $B2B_OUTDIR"
