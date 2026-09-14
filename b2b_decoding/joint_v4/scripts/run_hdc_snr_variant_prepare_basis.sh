#!/bin/bash
#SBATCH -J b2b_hdc_var_basis
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/b2b_hdc_var_basis_%j.out
#SBATCH -e logs/b2b_hdc_var_basis_%j.err
#
# Prepare passthrough basis for an HDC SNR variant. Requires B2B_OUTDIR
# (and matching feature-set env) via sbatch --export.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

: "${B2B_OUTDIR:?B2B_OUTDIR must be set at sbatch --export}"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
# Force outdir (do not inherit a stale login-shell B2B_OUTDIR).
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=observed
export B2B_EPOCH_ANCHOR=offset
export B2B_BASIS_MODE=independent
export B2B_PASSTHROUGH=1
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_NO_PLOT=1
export B2B_SPATIAL_DENOISE="${B2B_SPATIAL_DENOISE:-none}"
export B2B_SPATIAL_N_COMP="${B2B_SPATIAL_N_COMP:-40}"
export B2B_TIME_BIN_MS="${B2B_TIME_BIN_MS:-0}"
export B2B_TG_STRIDE="${B2B_TG_STRIDE:-2}"
export B2B_LAMBDA_TUNE_ALL="${B2B_LAMBDA_TUNE_ALL:-0}"
export B2B_STRICT_FEAT_DIMS="${B2B_STRICT_FEAT_DIMS:-1}"
export B2B_FEATURE_SETS="${B2B_FEATURE_SETS:-hdc_phonetic:hdc_word_form:hdc_lexical_syntactic:hdc_syntactic_operation:hdc_syntactic_state:hdc_semantic}"
export B2B_FAMILY_NAMES="${B2B_FAMILY_NAMES:-phonetic:word_form:lexical_syntactic:syntactic_operation:syntactic_state:semantic}"

module load julia/1.12.6

# Optional tagged feature regen (parsimony).
if [[ -n "${HDC_FEATURE_TAG:-}" ]]; then
  export HDC_MIN_POS="${HDC_MIN_POS:-50}"
  PY="${HDC_PYTHON:-}"
  if [[ -z "$PY" && -x .venv_gpt2/bin/python3 ]]; then PY=.venv_gpt2/bin/python3; fi
  if [[ -z "$PY" ]]; then PY=python3; fi
  echo "Regenerating tagged HDC features tag=${HDC_FEATURE_TAG} min_pos=${HDC_MIN_POS}"
  "$PY" scripts/prepare_hdc_features.py
fi

mkdir -p "${B2B_OUTDIR}" "${B2B_BASIS_DIR}"
# Never reuse another arm's manifest.
rm -f "${B2B_OUTDIR}/_analysis_manifest.json"

echo "Preparing basis → ${B2B_BASIS_DIR}"
echo "  out=${B2B_OUTDIR} spatial=${B2B_SPATIAL_DENOISE} bin=${B2B_TIME_BIN_MS} ltall=${B2B_LAMBDA_TUNE_ALL}"
echo "  features=${B2B_FEATURE_SETS}"
julia --project=.. prepare_feature_basis.jl
