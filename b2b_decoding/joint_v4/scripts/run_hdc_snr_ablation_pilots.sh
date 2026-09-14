#!/bin/bash
# SNR ablations on the HDC acoures arm (3-subject pilot each).
#
# Validates whether the DSS+bin score lift is from spatial, temporal, or both.
#
# Usage:
#   bash scripts/run_hdc_snr_ablation_pilots.sh
#   SUBJECTS="RN109 D007d D011d" bash scripts/run_hdc_snr_ablation_pilots.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
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
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntactic_operation,hdc_syntactic_state,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic

module load julia/1.12.6

SUBJECTS_STR="${SUBJECTS:-RN109 D007d D011d}"
read -r -a SUBJECTS_ARR <<< "$SUBJECTS_STR"

# name|spatial|n_comp|bin_ms
ABLATIONS=(
  "dss_only|dss|40|0"
  "bin_only|none|40|20"
  "pca40_bin20|pca|40|20"
)

# Reuse basis from the main SNR arm when features match (passthrough).
BASE_BASIS="${B2B_SNR_BASIS:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20/_basis}"

for spec in "${ABLATIONS[@]}"; do
  IFS='|' read -r name spatial ncomp bin <<< "$spec"
  export B2B_SPATIAL_DENOISE="$spatial"
  export B2B_SPATIAL_N_COMP="$ncomp"
  export B2B_TIME_BIN_MS="$bin"
  export B2B_OUTDIR="/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_${name}"
  export B2B_BASIS_DIR="${B2B_OUTDIR}/_basis"

  echo "======== ABLATION $name  spatial=$spatial bin=${bin}ms ========"
  if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
    mkdir -p "${B2B_BASIS_DIR}"
    if [[ -f "${BASE_BASIS}/feature_basis.npz" ]]; then
      cp -a "${BASE_BASIS}/." "${B2B_BASIS_DIR}/"
      echo "  copied basis from $BASE_BASIS"
    else
      julia --project=.. prepare_feature_basis.jl
    fi
  fi
  # Fresh outdir manifest for this fingerprint (basis copy may carry old manifest).
  rm -f "${B2B_OUTDIR}/_analysis_manifest.json"

  for p in "${SUBJECTS_ARR[@]}"; do
    echo "---- $name / $p ----"
    julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/hdc_ablation_${name}_${p}.out"
  done
done

echo "Ablation pilots done."
