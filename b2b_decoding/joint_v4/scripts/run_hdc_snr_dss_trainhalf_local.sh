#!/usr/bin/env bash
# Local parallel runner for DSS train_half observed arm (when Slurm is down).
# Usage: bash scripts/run_hdc_snr_dss_trainhalf_local.sh [concurrency]
set -euo pipefail
cd "$(dirname "$0")/.."
NJOBS="${1:-4}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-2}"
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
export B2B_DO_TG=0
export B2B_SPATIAL_DENOISE=dss
export B2B_SPATIAL_N_COMP=40
export B2B_TIME_BIN_MS=20
export B2B_DSS_FIT_SCOPE=train_half
export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntactic_operation,hdc_syntactic_state,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40tb_bin20}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"

module load julia/1.12.6
mkdir -p logs validation_cap_dss40_bin20/trainhalf_local

run_one() {
  local idx="$1"
  local part
  part=$(python3 - <<PY
import csv
rows=[r for r in csv.DictReader(open("cohort_groups.csv")) if r["include_primary"] in ("1","True","true")]
print(rows[${idx}-1]["participant"])
PY
)
  local out_csv="${B2B_OUTDIR}/${part}/${part}_b2b_family_agg.csv"
  if [[ -f "$out_csv" && "${B2B_FORCE:-0}" != "1" ]]; then
    echo "[skip] $idx $part"
    return 0
  fi
  echo "[start] $idx $part"
  if julia --project=/orcd/pool/005/haolun52/dyslexia_natualistics_listing/b2b_decoding \
      b2b_joint_v4_pipeline.jl "$idx" \
      > "logs/trainhalf_local_${idx}_${part}.out" 2>&1; then
    echo "[done] $idx $part"
  else
    echo "[FAIL] $idx $part" | tee -a validation_cap_dss40_bin20/trainhalf_local/failures.txt
    return 1
  fi
}
export -f run_one
export B2B_OUTDIR B2B_BASIS_DIR B2B_EXTRACTOR_DIR JULIA_DEPOT_PATH B2B_BLAS_THREADS
export B2B_MODE B2B_EPOCH_ANCHOR B2B_BASIS_MODE B2B_PASSTHROUGH B2B_PCA_K B2B_PCA_FIT_MAX
export B2B_DRIFT_KNOTS B2B_INCLUDE_NUISANCE B2B_ALLOW_ZERO_FEATURE_ROWS B2B_NO_PLOT B2B_DO_TG
export B2B_SPATIAL_DENOISE B2B_SPATIAL_N_COMP B2B_TIME_BIN_MS B2B_DSS_FIT_SCOPE
export B2B_FEATURE_SETS B2B_FAMILY_NAMES B2B_FORCE

seq 1 63 | xargs -P "$NJOBS" -I{} bash -c 'run_one "$@"' _ {}
echo "Local train_half sweep finished."
