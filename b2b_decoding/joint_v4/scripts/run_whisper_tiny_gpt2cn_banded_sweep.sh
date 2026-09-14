#!/usr/bin/env bash
# Sweep Tiny speech + GPT2-CN B2B across bands × anchors (local fallback).
set -euo pipefail
cd "$(dirname "$0")/.."

NJOBS="${1:-2}"
BANDS="${BANDS:-delta theta broadband alpha beta}"
ANCHORS="${ANCHORS:-onset offset}"
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-2}"

bash scripts/wire_banded_b2b_extractors.sh

LOG="logs/banded_tiny_gpt2cn_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
{
  echo "Sweep start $(date -Is)  NJOBS=${NJOBS} BLAS=${B2B_BLAS_THREADS}"
  echo "BANDS=${BANDS}"
  echo "ANCHORS=${ANCHORS}"
} | tee "$LOG"

for band in $BANDS; do
  for anchor in $ANCHORS; do
    echo "==== ${band} / ${anchor} ====" | tee -a "$LOG"
    bash scripts/run_whisper_tiny_gpt2cn_banded_local.sh "$band" "$anchor" "$NJOBS" \
      2>&1 | tee -a "$LOG"
  done
done

echo "Sweep finished $(date -Is)" | tee -a "$LOG"
echo "Log: $LOG"
export SCORE_COLS=mean_trace
bash scripts/compare_whisper_tiny_gpt2cn_banded.sh 2>&1 | tee -a "$LOG"
