#!/bin/bash
# Stage 2c tree-syntax HDC arm: replace syntax_proxy with
# syntactic_operation + syntactic_state from lppCN_tree.txt.
#
# Isolated outdir from the wordinfo-proxy pilot.
#
# Usage:
#   bash scripts/run_hdc_syntax_pilot.sh           # RN109 / D007d / D011d
#   B2B_DO_TG=1 bash scripts/run_hdc_syntax_pilot.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax}"
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

export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntactic_operation,hdc_syntactic_state,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic
export B2B_ALLOW_ZERO_FEATURE_ROWS=1

module load julia/1.12.6

echo "=== HDC B2B tree-syntax arm ==="
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
"$PY" scripts/prepare_hdc_syntax.py

if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "Preparing tree-syntax passthrough feature basis…"
  julia --project=.. prepare_feature_basis.jl
fi

SUBJECTS=("$@")
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  SUBJECTS=(RN109 D007d D011d)
fi

for p in "${SUBJECTS[@]}"; do
  echo "======== HDC SYNTAX $p ========"
  julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/hdc_syntax_${p}.out"
done

echo "HDC syntax arm complete → $B2B_OUTDIR"
