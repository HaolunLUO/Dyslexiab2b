#!/bin/bash
# HDC SNR pilot: DSS (40 comps) + 20 ms temporal binning on acoures zqc EEG.
#
# Isolated outdir from the baseline zqc arm. Defaults off for TG (smoke first).
#
# Usage:
#   bash scripts/run_hdc_snr_dss_bin_pilot.sh              # RN109 / D007d / D011d
#   B2B_DO_TG=1 bash scripts/run_hdc_snr_dss_bin_pilot.sh
#   B2B_SPATIAL_DENOISE=pca bash scripts/run_hdc_snr_dss_bin_pilot.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20}"
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
# Match array / outdir fingerprint (default 2; at 50 Hz effective → 25 Hz TG grid).
export B2B_TG_STRIDE="${B2B_TG_STRIDE:-2}"

# SNR knobs (must match prepare_basis / array fingerprints)
export B2B_SPATIAL_DENOISE="${B2B_SPATIAL_DENOISE:-dss}"
export B2B_SPATIAL_N_COMP="${B2B_SPATIAL_N_COMP:-40}"
export B2B_TIME_BIN_MS="${B2B_TIME_BIN_MS:-20}"

export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntactic_operation,hdc_syntactic_state,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic
export B2B_ALLOW_ZERO_FEATURE_ROWS=1

module load julia/1.12.6

echo "=== HDC B2B SNR arm (DSS + temporal bin) ==="
echo "  extractor : $B2B_EXTRACTOR_DIR"
echo "  outdir    : $B2B_OUTDIR"
echo "  spatial   : $B2B_SPATIAL_DENOISE  n_comp=$B2B_SPATIAL_N_COMP"
echo "  time_bin  : ${B2B_TIME_BIN_MS} ms"
echo "  TG        : $B2B_DO_TG  stride=$B2B_TG_STRIDE"

test -d "$B2B_EXTRACTOR_DIR" || {
  echo "Missing residual extractor; run scripts/regress_out_acoustic_mne.py first"
  exit 1
}

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
  echo "Preparing SNR-arm passthrough feature basis…"
  julia --project=.. prepare_feature_basis.jl
fi

SUBJECTS=("$@")
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  SUBJECTS=(RN109 D007d D011d)
fi

for p in "${SUBJECTS[@]}"; do
  echo "======== HDC SNR DSS+BIN $p ========"
  julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/hdc_snr_dss_bin_${p}.out"
done

echo "HDC SNR DSS+bin arm complete → $B2B_OUTDIR"
