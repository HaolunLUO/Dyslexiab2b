#!/bin/bash
# Group comparison for MFA v2 phone / tone / envelope B2B (onset-locked passthrough).
#
# Primary score: mean_score = trace/K. mean_trace retained only as sensitivity.
#
# Usage:
#   bash scripts/run_mfa_phone_tone_env_group_comparison.sh
#   PLOT_ONLY=1 bash scripts/run_mfa_phone_tone_env_group_comparison.sh
#   SCORE_COL=mean_trace bash scripts/run_mfa_phone_tone_env_group_comparison.sh  # sensitivity
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_mfa_v2_phone_tone_env_onset_passthrough}"
OUT_DIR="${1:-group_comparison_b2b_mfa_v2_phone_tone_env_onset_passthrough}"
VENV="${B2B_PY_VENV:-$(pwd)/.venv_gpt2}"
PYTHON="${PYTHON:-${VENV}/bin/python}"
FAMILIES="${B2B_FAMILY_NAMES:-envelope,phones,tone}"
SCORE_COL="${SCORE_COL:-mean_score}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON" >&2
  exit 1
fi

if ! "$PYTHON" -c "import numpy, pandas, scipy, matplotlib" 2>/dev/null; then
  echo "Installing scipy/matplotlib into $VENV"
  "$PYTHON" -m pip install -q scipy matplotlib
fi

n_agg=$(find "$RESULTS_DIR" -mindepth 2 -maxdepth 2 -name '*_b2b_family_agg.csv' | wc -l)
echo "Using $($PYTHON --version) at $PYTHON"
echo "  results : $RESULTS_DIR  ($n_agg family_agg files)"
echo "  outdir  : $OUT_DIR"
echo "  families: $FAMILIES"
echo "  score   : $SCORE_COL"
[[ "$n_agg" -eq 63 ]] || {
  echo "ERROR: expected 63 family_agg files, got $n_agg" >&2
  exit 1
}

SUMMARY="$OUT_DIR/group_comparison_summary.csv"
if [[ "${PLOT_ONLY:-0}" != "1" ]]; then
  "$PYTHON" compare_b2b_groups.py \
    --results-dir "$RESULTS_DIR" \
    --groups cohort_groups.csv \
    --out-dir "$OUT_DIR" \
    --families "$FAMILIES" \
    --score-col "$SCORE_COL" \
    --skip-existence-if-no-nulls
  echo "Wrote $SUMMARY"
elif [[ ! -f "$SUMMARY" ]]; then
  echo "PLOT_ONLY=1 but missing $SUMMARY" >&2
  exit 1
fi

"$PYTHON" plot_group_comparison.py \
  --in-dir "$OUT_DIR" \
  --families "$FAMILIES" \
  --title "MFA v2 envelope / phones / tone B2B · onset · ${SCORE_COL} · group mean ± SE · confirm 0–0.8 s shaded"
echo "Wrote $OUT_DIR/plots/group_means_by_family.png"
