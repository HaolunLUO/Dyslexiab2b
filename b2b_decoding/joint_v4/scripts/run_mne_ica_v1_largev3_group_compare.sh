#!/usr/bin/env bash
# After MNE-ICA v1 observed+nulls finish: aggregate leftover chunks and group-compare.
#
#   B2B_EPOCH_ANCHOR=onset bash scripts/run_mne_ica_v1_largev3_group_compare.sh
#   B2B_EPOCH_ANCHOR=offset bash scripts/run_mne_ica_v1_largev3_group_compare.sh
set -euo pipefail
cd "$(dirname "$0")/.."

POOL="${POOL:-/orcd/pool/005/haolun52}"
PY="${PY:-${POOL}/dyslexia_natualistics_listing/b2b_decoding/joint_v4/.venv_gpt2/bin/python}"
ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"

if [[ "$ANCHOR" == "offset" ]]; then
  RES="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_offset_joint_k8_mne_ica_v1}"
  OUT="group_comparison_b2b_largev3_offset_joint_k8_mne_ica_v1"
  TITLE="Whisper large-v3 B2B offset · MNE-ICA v1 · group mean ± SE · confirm 0–0.8 s shaded"
else
  RES="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_onset_joint_k8_mne_ica_v1}"
  OUT="group_comparison_b2b_largev3_onset_joint_k8_mne_ica_v1"
  TITLE="Whisper large-v3 B2B · MNE-ICA v1 · group mean ± SE · confirm 0–0.8 s shaded"
fi

n_agg=$(find "$RES" -mindepth 2 -maxdepth 2 -name '*_b2b_family_agg.csv' 2>/dev/null | wc -l)
echo "observed family_agg: ${n_agg}/63  dir=${RES}"
[[ "$n_agg" -ge 63 ]] || { echo "Observed incomplete"; exit 1; }

module load julia/1.12.6
B2B_OUTDIR="$RES" julia --project=.. aggregate_v4.jl --nulls-only || true

n_null=$(find "$RES" -mindepth 2 -maxdepth 2 -name '*_null_traces.csv' 2>/dev/null | wc -l)
echo "aggregated null_traces: ${n_null}/63"
[[ "$n_null" -ge 63 ]] || { echo "Nulls incomplete (${n_null}/63)"; exit 1; }

"$PY" compare_b2b_groups.py \
  --results-dir "$RES" \
  --groups cohort_groups.csv \
  --out-dir "$OUT"

if [[ -f plot_group_comparison.py ]]; then
  "$PY" plot_group_comparison.py --in-dir "$OUT" --title "$TITLE" || true
fi

echo "Done: ${OUT}"
