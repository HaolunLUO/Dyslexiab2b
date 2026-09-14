#!/bin/bash
#SBATCH -J b2b_hdc_acr_zqc_basis
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/b2b_hdc_acr_zqc_basis_%j.out
#SBATCH -e logs/b2b_hdc_acr_zqc_basis_%j.err
#
# HDC syntax features + passthrough basis on acoures EEG (zscore-only epoch QC).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc}"
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
export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntactic_operation,hdc_syntactic_state,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic

module load julia/1.12.6

test -d "$B2B_EXTRACTOR_DIR" || { echo "Missing residual extractor"; exit 1; }

PY="${HDC_PYTHON:-}"
if [[ -z "$PY" && -x .venv_gpt2/bin/python3 ]]; then PY=.venv_gpt2/bin/python3; fi
if [[ -z "$PY" ]]; then PY=python3; fi
"$PY" scripts/prepare_hdc_features.py
"$PY" scripts/prepare_hdc_syntax.py

echo "Preparing HDC acoures zqc basis → ${B2B_BASIS_DIR}"
julia --project=.. prepare_feature_basis.jl
