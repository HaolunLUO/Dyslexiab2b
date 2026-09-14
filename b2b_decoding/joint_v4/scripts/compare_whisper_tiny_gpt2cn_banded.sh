#!/usr/bin/env bash
# Group-compare Tiny speech + GPT2-CN banded arms (paper score Σ diag(H)).
set -euo pipefail
cd "$(dirname "$0")/.."

POOL="${POOL:-/orcd/pool/005/haolun52}"
BANDS="${BANDS:-delta theta broadband alpha beta}"
ANCHORS="${ANCHORS:-onset offset}"
VENV="${VENV:-$(pwd)/.venv_gpt2}"
PY="${PY:-${VENV}/bin/python}"
SCORE_COLS="${SCORE_COLS:-mean_trace}"
test -x "$PY" || { echo "Missing $PY"; exit 1; }

for band in $BANDS; do
  for anchor in $ANCHORS; do
    res="${POOL}/encoding_results_b2b_tinyspeech_gpt2cn_${anchor}_${band}_zqc_k8_reseg"
    n=$(find "$res" -name '*_b2b_family_agg.csv' 2>/dev/null | wc -l || true)
    if [[ "${n}" -lt 50 ]]; then
      echo "[skip] ${band}/${anchor}: only ${n}/63 → ${res}"
      continue
    fi
    for score in $SCORE_COLS; do
      out="group_comparison_b2b_tinyspeech_gpt2cn_${anchor}_${band}_zqc_k8_reseg_${score}"
      echo "==== compare ${band}/${anchor} ${score} n=${n} → ${out} ===="
      "$PY" compare_b2b_groups.py \
        --results-dir "$res" \
        --groups cohort_groups.csv \
        --out-dir "$out" \
        --families speech,gpt2cn \
        --score-col "$score" \
        --skip-existence-if-no-nulls \
        || echo "[warn] compare failed ${band}/${anchor}/${score}"
      if [[ -f "$out/group_comparison_summary.csv" ]]; then
        "$PY" plot_group_comparison.py \
          --in-dir "$out" \
          --families speech,gpt2cn \
          --title "Tiny speech + GPT2-CN ${band} ${anchor} (${score})" \
          || true
      fi
    done
  done
done
