#!/usr/bin/env bash
# Re-run Tiny speech+lang B2B + group analysis on freshly segmented banded EEG.
# Writes to *_zqc_k8_reseg so the previous banded run is left intact.
set -euo pipefail
cd "$(dirname "$0")/.."

NJOBS="${1:-2}"
export B2B_OUT_SUFFIX="${B2B_OUT_SUFFIX:-_reseg}"
export B2B_FORCE="${B2B_FORCE:-1}"
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-2}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export SCORE_COLS="${SCORE_COLS:-mean_trace}"

echo "=== reseg start $(date -Is) suffix=${B2B_OUT_SUFFIX} jobs=${NJOBS} ==="
bash scripts/wire_banded_b2b_extractors.sh
bash scripts/run_whisper_tiny_speechlang_banded_sweep.sh "$NJOBS"
echo "=== group compare $(date -Is) score=${SCORE_COLS} ==="
bash scripts/compare_whisper_tiny_speechlang_banded.sh
echo "=== reseg finished $(date -Is) ==="
