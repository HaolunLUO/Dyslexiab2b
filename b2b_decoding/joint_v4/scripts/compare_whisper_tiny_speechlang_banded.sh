#!/usr/bin/env bash
# Group-compare Tiny speech+lang banded arms (TD vs dyslexia).
#
# Usage:
#   bash scripts/compare_whisper_tiny_speechlang_banded.sh
#   BANDS="delta theta" ANCHORS="onset offset" bash scripts/...
set -euo pipefail
cd "$(dirname "$0")/.."

POOL="${POOL:-/orcd/pool/005/haolun52}"
BANDS="${BANDS:-delta theta broadband alpha beta}"
ANCHORS="${ANCHORS:-onset offset}"
VENV="${VENV:-$(cd "$(dirname "$0")/.." && pwd)/.venv_gpt2}"
PY="${PY:-${VENV}/bin/python}"
SCORE_COLS="${SCORE_COLS:-mean_score mean_z}"
test -x "$PY" || { echo "Missing $PY — set PY=..."; exit 1; }

for band in $BANDS; do
  for anchor in $ANCHORS; do
    res="${POOL}/encoding_results_b2b_whispertiny_${anchor}_speechlang_${band}_zqc_k8${B2B_OUT_SUFFIX:-}"
    n=$(find "$res" -name '*_b2b_family_agg.csv' 2>/dev/null | wc -l || true)
    if [[ "${n}" -lt 50 ]]; then
      echo "[skip] ${band}/${anchor}: only ${n}/63 done → ${res}"
      continue
    fi
    for score in $SCORE_COLS; do
      out="group_comparison_b2b_whispertiny_${anchor}_speechlang_${band}_zqc_k8${B2B_OUT_SUFFIX:-}_${score}"
      echo "==== compare ${band}/${anchor} score=${score} n=${n} → ${out} ===="
      "$PY" compare_b2b_groups.py \
        --results-dir "$res" \
        --groups cohort_groups.csv \
        --out-dir "$out" \
        --families speech,language \
        --score-col "$score" \
        --skip-existence-if-no-nulls \
        ${EXTRA_COMPARE_ARGS:-} || echo "[warn] compare failed ${band}/${anchor}/${score}"
      if [[ -f "$out/group_comparison_summary.csv" ]]; then
        "$PY" plot_group_comparison.py \
          --in-dir "$out" \
          --families speech,language \
          --title "Tiny speech+lang ${band} ${anchor} (${score}) · group mean ± SE" \
          || echo "[warn] plot failed ${band}/${anchor}/${score}"
      fi
    done
  done
done
