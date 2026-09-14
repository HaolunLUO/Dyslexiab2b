#!/bin/bash
# Local full-cohort observed run for lexical B2B (no Slurm).
# Usage: bash scripts/run_lexical_cohort_local.sh [N_PARALLEL]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

NPAR="${1:-8}"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_lexical_onset_dur_freq_surp_k1}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=observed
export B2B_EPOCH_ANCHOR=onset
export B2B_BASIS_MODE=independent
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_NO_PLOT=1
export B2B_BLAS_THREADS=4
export B2B_FEAT_ACOUSTIC=lexical_duration
export B2B_FEAT_SPEECH=lexical_frequency
export B2B_FEAT_LANGUAGE=gpt2cn_surprisal
export B2B_FAMILY_NAMES=duration,frequency,surprisal

module load julia/1.12.6

test -f "${B2B_BASIS_DIR}/feature_basis.npz" || {
  echo "Missing basis; run prepare_feature_basis / lexical pilot first"
  exit 1
}

mapfile -t PARTS < <(awk -F, 'NR>1 && $3==1 {print $1}' cohort_groups.csv)
echo "Lexical cohort: ${#PARTS[@]} participants, parallel=$NPAR"
echo "Outdir: $B2B_OUTDIR"
(( ${#PARTS[@]} == 63 )) || { echo "Expected 63 participants, got ${#PARTS[@]}"; exit 1; }

run_one() {
  local idx="$1"
  local p="$2"
  local agg="$B2B_OUTDIR/$p/${p}_b2b_family_agg.csv"
  if [[ -f "$agg" ]]; then
    echo "[skip] $idx $p"
    return 0
  fi
  echo "[start] $(date -Is) $idx $p"
  if julia --project=.. b2b_joint_v4_pipeline.jl "$idx" \
      >"logs/lexical_obs_${p}.out" 2>"logs/lexical_obs_${p}.err"; then
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
export B2B_FEAT_ACOUSTIC B2B_FEAT_SPEECH B2B_FEAT_LANGUAGE B2B_FAMILY_NAMES

# Build idx,participant lines then parallelize
: > logs/lexical_cohort_jobs.txt
for i in "${!PARTS[@]}"; do
  echo "$((i+1)) ${PARTS[$i]}" >> logs/lexical_cohort_jobs.txt
done

# Prefer GNU parallel if available; else xargs
if command -v parallel >/dev/null 2>&1; then
  parallel -j "$NPAR" --colsep ' ' run_one {1} {2} :::: logs/lexical_cohort_jobs.txt
else
  # bash job pool
  fail=0
  while read -r idx p; do
    while (( $(jobs -rp | wc -l) >= NPAR )); do
      wait -n || fail=$((fail+1))
    done
    run_one "$idx" "$p" &
  done < logs/lexical_cohort_jobs.txt
  while (( $(jobs -rp | wc -l) > 0 )); do
    wait -n || fail=$((fail+1))
  done
fi

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
echo "All 63 lexical observed fits present."
