#!/usr/bin/env bash
# Group comparison for ICA v1 envelope / pitch / onset / frequency / surprisal B2B.
#
#   bash scripts/run_mne_ica_v1_envpitch_group_compare.sh
#   B2B_EPOCH_ANCHOR=offset bash scripts/run_mne_ica_v1_envpitch_group_compare.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Login nodes often have RLIMIT_NPROC=64; OpenBLAS defaulting to 64 threads
# exhausts the process table and segfaults numpy import.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

POOL="${POOL:-/orcd/pool/005/haolun52}"
PY="${PY:-$(pwd)/.venv_gpt2/bin/python}"
ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
if [[ -n "${B2B_OUTDIR:-}" ]]; then
  RES="$B2B_OUTDIR"
  OUT="${OUT_DIR:-group_comparison_$(basename "$RES")}"
  TITLE="${PLOT_TITLE:-ICA v1 B2B ${ANCHOR} · confirm 0–0.8 s}"
elif [[ "$ANCHOR" == "offset" ]]; then
  RES="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_offset_freq_surp_passthrough"
  OUT="${OUT_DIR:-group_comparison_b2b_mne_ica_v1_envpitch_offset_freq_surp_passthrough}"
  TITLE="ICA v1 B2B offset · envelope / pitch / onset / frequency / surprisal · confirm 0–0.8 s"
else
  RES="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_onset_freq_surp_passthrough"
  OUT="${OUT_DIR:-group_comparison_b2b_mne_ica_v1_envpitch_onset_freq_surp_passthrough}"
  TITLE="ICA v1 B2B · envelope / pitch / onset / frequency / surprisal · confirm 0–0.8 s"
fi
FAMILIES="${B2B_FAMILY_NAMES:-envelope,pitch,onset,frequency,surprisal}"
FAMILIES="${FAMILIES//:/,}"
SCORE_COL="${SCORE_COL:-mean_trace}"

n_agg=$(find "$RES" -mindepth 2 -maxdepth 2 -name '*_b2b_family_agg.csv' 2>/dev/null | wc -l)
echo "observed family_agg: ${n_agg}/63  dir=${RES}  anchor=${ANCHOR}"
[[ "$n_agg" -ge 63 ]] || { echo "Observed incomplete"; exit 1; }

SKIP=()
n_null=$(find "$RES" -mindepth 2 -maxdepth 2 -name '*_null_traces.csv' 2>/dev/null | wc -l)
if [[ "$n_null" -lt 63 ]]; then
  echo "Nulls incomplete (${n_null}/63); running observed-only group contrast"
  SKIP+=(--skip-existence-if-no-nulls)
fi

"$PY" compare_b2b_groups.py \
  --results-dir "$RES" \
  --groups cohort_groups.csv \
  --out-dir "$OUT" \
  --families "$FAMILIES" \
  --score-col "$SCORE_COL" \
  "${SKIP[@]}"

"$PY" plot_group_comparison.py \
  --in-dir "$OUT" \
  --families "$FAMILIES" \
  --title "$TITLE" \
  --xlabel "${PLOT_XLABEL:-Time (s, ${ANCHOR}-locked)}"

echo "Done: ${OUT}"
