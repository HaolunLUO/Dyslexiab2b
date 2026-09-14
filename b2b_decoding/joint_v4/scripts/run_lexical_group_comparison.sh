#!/bin/bash
# Group comparison for lexical B2B (duration / frequency / surprisal).
# Login-node `python3` is often 3.6 and has no numpy. Use the existing 3.10 venv.
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_lexical_onset_dur_freq_surp_k1}"
OUT_DIR="${1:-group_comparison_b2b_lexical_onset_dur_freq_surp_k1}"
VENV="${B2B_PY_VENV:-$(pwd)/.venv_gpt2}"
PYTHON="${PYTHON:-${VENV}/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON" >&2
  echo "Create it with: module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64" >&2
  echo "  python3 -m venv .venv_gpt2 && .venv_gpt2/bin/pip install numpy pandas scipy" >&2
  exit 1
fi

if ! "$PYTHON" -c "import numpy, pandas, scipy, matplotlib" 2>/dev/null; then
  echo "Installing scipy/matplotlib into $VENV"
  "$PYTHON" -m pip install -q scipy matplotlib
fi

echo "Using $($PYTHON --version) at $PYTHON"

SUMMARY="$OUT_DIR/group_comparison_summary.csv"
if [[ "${PLOT_ONLY:-0}" != "1" ]]; then
  "$PYTHON" compare_b2b_groups.py \
    --results-dir "$RESULTS_DIR" \
    --groups cohort_groups.csv \
    --out-dir "$OUT_DIR" \
    --families duration,frequency,surprisal \
    --skip-existence-if-no-nulls
  echo "Wrote $SUMMARY"
elif [[ ! -f "$SUMMARY" ]]; then
  echo "PLOT_ONLY=1 but missing $SUMMARY" >&2
  exit 1
fi

"$PYTHON" plot_group_comparison.py \
  --in-dir "$OUT_DIR" \
  --families duration,frequency,surprisal
echo "Wrote $OUT_DIR/plots/group_means_by_family.png"
