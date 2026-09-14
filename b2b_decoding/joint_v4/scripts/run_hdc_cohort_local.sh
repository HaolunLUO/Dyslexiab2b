#!/bin/bash
# Local full-cohort observed run for HDC B2B (no Slurm).
# Login-node user cgroup is ~8–10 GB — keep N_PARALLEL at 1 (2 max).
# Prefer: B2B_DO_TG=1 sbatch scripts/run_hdc_observed_array.sh
# Usage: bash scripts/run_hdc_cohort_local.sh [N_PARALLEL]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

NPAR="${1:-1}"

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
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_NO_PLOT=1
export B2B_BLAS_THREADS=4
export B2B_DO_TG="${B2B_DO_TG:-0}"
export B2B_TG_STRIDE="${B2B_TG_STRIDE:-2}"
export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntax_proxy,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntax_proxy,semantic

module load julia/1.12.6

test -f "${B2B_BASIS_DIR}/feature_basis.npz" || {
  echo "Missing basis; run scripts/run_hdc_pilot.sh first"
  exit 1
}

mapfile -t PARTS < <(awk -F, 'NR>1 && $3==1 {print $1}' cohort_groups.csv)
echo "HDC cohort: ${#PARTS[@]} participants, parallel=$NPAR  TG=$B2B_DO_TG"
(( ${#PARTS[@]} == 63 )) || { echo "Expected 63 participants, got ${#PARTS[@]}"; exit 1; }

run_one() {
  local idx="$1"
  local p="$2"
  local agg="$B2B_OUTDIR/$p/${p}_b2b_family_agg.csv"
  if [[ -f "$agg" && "${B2B_FORCE:-0}" != "1" ]]; then
    if [[ "${B2B_DO_TG}" == "1" && ! -f "$B2B_OUTDIR/$p/${p}_tg_metrics.csv" ]]; then
      echo "[tg-only] $idx $p"
    else
      echo "[skip] $idx $p"
      return 0
    fi
  fi
  echo "[start] $(date -Is) $idx $p"
  if stdbuf -oL -eL julia --project=.. b2b_joint_v4_pipeline.jl "$idx" \
      >"logs/hdc_obs_${p}.out" 2>"logs/hdc_obs_${p}.err"; then
    echo "[done]  $(date -Is) $idx $p"
    return 0
  else
    echo "[FAIL]  $(date -Is) $idx $p"
    return 1
  fi
}
export -f run_one
export B2B_OUTDIR JULIA_DEPOT_PATH B2B_EXTRACTOR_DIR B2B_BASIS_DIR
export B2B_MODE B2B_EPOCH_ANCHOR B2B_BASIS_MODE B2B_PCA_K B2B_PCA_FIT_MAX
export B2B_DRIFT_KNOTS B2B_INCLUDE_NUISANCE B2B_NO_PLOT B2B_BLAS_THREADS
export B2B_PASSTHROUGH B2B_FEATURE_SETS B2B_FAMILY_NAMES B2B_ALLOW_ZERO_FEATURE_ROWS
export B2B_DO_TG B2B_TG_STRIDE B2B_FORCE

: > logs/hdc_cohort_jobs.txt
for i in "${!PARTS[@]}"; do
  echo "$((i+1)) ${PARTS[$i]}" >> logs/hdc_cohort_jobs.txt
done

fail=0
while read -r idx p; do
  while (( $(jobs -rp | wc -l) >= NPAR )); do
    wait -n || fail=$((fail+1))
  done
  run_one "$idx" "$p" &
done < logs/hdc_cohort_jobs.txt
while (( $(jobs -rp | wc -l) > 0 )); do
  wait -n || fail=$((fail+1))
done

ok=0
missing=()
for p in "${PARTS[@]}"; do
  if [[ -f "$B2B_OUTDIR/$p/${p}_b2b_family_agg.csv" ]]; then
    ok=$((ok+1))
  else
    missing+=("$p")
  fi
done
echo "Complete: $ok / ${#PARTS[@]}"
if (( ${#missing[@]} > 0 )); then
  echo "Missing: ${missing[*]}"
  exit 1
fi
echo "All 63 HDC observed fits present."
