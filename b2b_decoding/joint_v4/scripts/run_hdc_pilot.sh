#!/bin/bash
# HDC pilot: offset-locked joint B2B on wordinfo + GloVe PCA-10 families.
#
# Families (K = n_cols, no PCA, no drift):
#   phonetic, word_form, lexical_syntactic, syntax_proxy, semantic
#
# Usage:
#   bash scripts/run_hdc_pilot.sh           # RN109 / D007d / D011d
#   bash scripts/run_hdc_pilot.sh RN109
#   B2B_DO_TG=1 bash scripts/run_hdc_pilot.sh   # also write TG matrices
#
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_pilot}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"

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
export B2B_DO_TG="${B2B_DO_TG:-0}"
export B2B_TG_STRIDE="${B2B_TG_STRIDE:-2}"

export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntax_proxy,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntax_proxy,semantic
export B2B_ALLOW_ZERO_FEATURE_ROWS=1

module load julia/1.12.6

echo "=== HDC B2B pilot ==="
echo "  outdir   : $B2B_OUTDIR"
echo "  features : $B2B_FEATURE_SETS"
echo "  families : $B2B_FAMILY_NAMES"
echo "  anchor   : $B2B_EPOCH_ANCHOR  passthrough=$B2B_PASSTHROUGH  TG=$B2B_DO_TG"

PY="${HDC_PYTHON:-}"
if [[ -z "$PY" && -x .venv_gpt2/bin/python3 ]]; then
  PY=.venv_gpt2/bin/python3
fi
if [[ -z "$PY" ]]; then
  PY=python3
fi
"$PY" scripts/prepare_hdc_features.py

if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "Preparing HDC passthrough feature basis…"
  julia --project=.. prepare_feature_basis.jl
fi

SUBJECTS=("$@")
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  SUBJECTS=(RN109 D007d D011d)
fi

for p in "${SUBJECTS[@]}"; do
  echo "======== HDC PILOT $p ========"
  julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/hdc_pilot_${p}.out"
done

echo "HDC pilot complete → $B2B_OUTDIR"
echo "Analyze with: $PY scripts/analyze_hdc_pilot.py --results-dir $B2B_OUTDIR"
