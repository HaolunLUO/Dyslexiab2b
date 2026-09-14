#!/bin/bash
# Feature-parsimony + all-time λ tune on top of DSS40+bin20 (pilot).
#
# Raises HDC_MIN_POS to drop rare POS one-hots; writes tagged feature files
# (*_mp50) so the baseline HDC features are untouched.
#
# Usage:
#   bash scripts/run_hdc_snr_parsimony_pilot.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_EXTRACTOR_DIR="/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual"
# Force outdir — do not inherit a stale login-shell B2B_OUTDIR.
export B2B_OUTDIR="/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20_mp50_ltall"
export B2B_BASIS_DIR="${B2B_OUTDIR}/_basis"

export B2B_MODE=observed
export B2B_EPOCH_ANCHOR=offset
export B2B_BASIS_MODE=independent
export B2B_PASSTHROUGH=1
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_NO_PLOT=1
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-4}"
export B2B_DO_TG=0
export B2B_TG_STRIDE=2
export B2B_SPATIAL_DENOISE=dss
export B2B_SPATIAL_N_COMP=40
export B2B_TIME_BIN_MS=20
export B2B_LAMBDA_TUNE_ALL=1
export B2B_STRICT_FEAT_DIMS=0
export B2B_ALLOW_ZERO_FEATURE_ROWS=1

export HDC_MIN_POS="${HDC_MIN_POS:-50}"
export HDC_FEATURE_TAG="${HDC_FEATURE_TAG:-mp50}"
export B2B_FEATURE_SETS=hdc_phonetic_mp50,hdc_word_form_mp50,hdc_lexical_syntactic_mp50,hdc_syntactic_operation,hdc_syntactic_state,hdc_semantic_mp50
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic

module load julia/1.12.6

PY="${HDC_PYTHON:-}"
if [[ -z "$PY" && -x .venv_gpt2/bin/python3 ]]; then PY=.venv_gpt2/bin/python3; fi
if [[ -z "$PY" ]]; then PY=python3; fi

echo "=== HDC parsimony+λtune pilot  min_pos=$HDC_MIN_POS tag=$HDC_FEATURE_TAG ==="
"$PY" scripts/prepare_hdc_features.py
# Syntax tree features unchanged (already dense).
test -f "${B2B_EXTRACTOR_DIR}/_shared_wordlocked_features/section_001/X_word_hdc_syntactic_operation.npy"

if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  julia --project=.. prepare_feature_basis.jl
fi

SUBJECTS=("$@")
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  SUBJECTS=(RN109 D007d D011d)
fi
for p in "${SUBJECTS[@]}"; do
  echo "======== PARSIMONY $p ========"
  julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/hdc_parsimony_${p}.out"
done
echo "Parsimony pilot → $B2B_OUTDIR"
