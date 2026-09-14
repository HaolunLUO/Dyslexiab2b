#!/usr/bin/env bash
# Sweep Tiny speech+lang B2B across bands × anchors (local; Slurm down).
#
# Compute plan (from prior Tiny arrays + current login-node load):
#   - Prior Tiny zQC array (%20) finished 63 subjects in ~7 min wall → ~2–3 min/subj
#   - Prior Tiny DSS+bin array (%16) ~11 min span
#   - Login node ulimit -u=768; P=6 died with EAGAIN (no Julia threads left)
#   - Measured local pilot: ~219 s/subj (BLAS=2, no DSS). Keep P=2.
#   - No DSS/bin on banded arms (bands already filter; keep apples-to-apples)
#   - Reuse existing Tiny feature bases (no per-band basis rebuild)
#   - Priority order: delta → theta → broadband → alpha → beta; onset then offset
#
# Usage:
#   bash scripts/run_whisper_tiny_speechlang_banded_sweep.sh [concurrency]
#   BANDS="delta theta" ANCHORS="onset offset" bash scripts/...
set -euo pipefail
cd "$(dirname "$0")/.."

NJOBS="${1:-2}"
BANDS="${BANDS:-delta theta broadband alpha beta}"
ANCHORS="${ANCHORS:-onset offset}"
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-2}"

bash scripts/wire_banded_b2b_extractors.sh

LOG="logs/banded_tiny_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
{
  echo "Sweep start $(date -Is)  NJOBS=${NJOBS} BLAS=${B2B_BLAS_THREADS}"
  echo "BANDS=${BANDS}"
  echo "ANCHORS=${ANCHORS}"
} | tee "$LOG"

for band in $BANDS; do
  for anchor in $ANCHORS; do
    echo "==== ${band} / ${anchor} ====" | tee -a "$LOG"
    bash scripts/run_whisper_tiny_speechlang_banded_local.sh "$band" "$anchor" "$NJOBS" \
      2>&1 | tee -a "$LOG"
  done
done

echo "Sweep finished $(date -Is)" | tee -a "$LOG"
echo "Log: $LOG"
