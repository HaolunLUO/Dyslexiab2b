#!/bin/bash
# Lexical King-style B2B: onset-locked EEG × (duration, frequency, GPT2CN surprisal)
#
# Families are three scalar predictors (K=1 each), joint in one B2B H.
# Outdir is isolated from Whisper Large-v2 primary.
#
# Usage:
#   bash scripts/run_lexical_pilot.sh           # RN109 / D007d / D011d
#   bash scripts/run_lexical_pilot.sh RN109     # single subject

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_lexical_onset_dur_freq_surp_k1}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"

export B2B_MODE=observed
export B2B_EPOCH_ANCHOR=onset
export B2B_BASIS_MODE=independent
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS="${B2B_DRIFT_KNOTS:-0}"   # scalars: skip slow-drift RBF (0 = off)
export B2B_INCLUDE_NUISANCE=0
export B2B_NO_PLOT=1
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-4}"

# Reuse the three feature slots as lexical scalars
export B2B_FEAT_ACOUSTIC=lexical_duration
export B2B_FEAT_SPEECH=lexical_frequency
export B2B_FEAT_LANGUAGE=gpt2cn_surprisal
export B2B_FAMILY_NAMES=duration,frequency,surprisal

module load julia/1.12.6

echo "=== Lexical B2B pilot ==="
echo "  outdir   : $B2B_OUTDIR"
echo "  features : $B2B_FEAT_ACOUSTIC | $B2B_FEAT_SPEECH | $B2B_FEAT_LANGUAGE"
echo "  families : $B2B_FAMILY_NAMES"
echo "  K        : $B2B_PCA_K  anchor=$B2B_EPOCH_ANCHOR"

# 1) Ensure GPT2CN surprisal + lexical duration/frequency exist
SURP1="$B2B_EXTRACTOR_DIR/_shared_wordlocked_features/section_001/X_word_gpt2cn_surprisal.npy"
if [[ ! -f "$SURP1" ]]; then
  echo "ERROR: missing $SURP1"
  echo "Compute surprisal first, e.g.:"
  echo "  module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64"
  echo "  python3 -m venv .venv_gpt2 && . .venv_gpt2/bin/activate"
  echo "  pip install torch transformers numpy pandas"
  echo "  python scripts/compute_gpt2cn_surprisal.py \\"
  echo "    --timing-root \$B2B_EXTRACTOR_DIR/_shared_wordlocked_features \\"
  echo "    --out-root    \$B2B_EXTRACTOR_DIR/_shared_wordlocked_features \\"
  echo "    --sections 1 2"
  exit 1
fi

julia --project=.. scripts/prepare_lexical_features.jl

if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "Preparing lexical feature basis…"
  julia --project=.. prepare_feature_basis.jl
fi

SUBJECTS=("$@")
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  SUBJECTS=(RN109 D007d D011d)
fi

for p in "${SUBJECTS[@]}"; do
  echo "======== LEXICAL PILOT $p ========"
  julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/lexical_pilot_${p}.out"
done

echo "Lexical pilot complete → $B2B_OUTDIR"
