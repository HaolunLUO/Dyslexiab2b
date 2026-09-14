#!/bin/bash
# Re-run group comparison (+ plots) after null_traces exist for both anchors.
#
# Usage:
#   bash scripts/run_whisperlargev3_group_compare.sh
#   B2B_EPOCH_ANCHOR=offset bash scripts/run_whisperlargev3_group_compare.sh
set -euo pipefail
cd "$(dirname "$0")/.."

POOL="${POOL:-/orcd/pool/005/haolun52}"
PY="${PY:-${POOL}/dyslexia_natualistics_listing/b2b_decoding/joint_v4/.venv_gpt2/bin/python}"
ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"

if [[ "$ANCHOR" == "offset" ]]; then
  RES="${POOL}/encoding_results_b2b_largev3_offset_joint_k8"
  OUT="group_comparison_b2b_largev3_offset_joint_k8"
  TITLE="Whisper large-v3 B2B offset · group mean ± SE · confirm 0–0.8 s shaded"
else
  RES="${POOL}/encoding_results_b2b_largev3_onset_joint_k8"
  OUT="group_comparison_b2b_largev3_onset_joint_k8"
  TITLE="Whisper large-v3 B2B · group mean ± SE · confirm 0–0.8 s shaded"
fi

# Require at least one aggregated null file
sample=$(ls -d "${RES}"/*/ 2>/dev/null | head -1)
sample=${sample%/}
sample=${sample##*/}
if [[ -z "$sample" ]] || [[ ! -f "${RES}/${sample}/${sample}_null_traces.csv" ]]; then
  echo "Missing aggregated null_traces under ${RES}; wait for null array to finish."
  exit 1
fi

"$PY" compare_b2b_groups.py \
  --results-dir "$RES" \
  --groups cohort_groups.csv \
  --out-dir "$OUT"

"$PY" plot_group_comparison.py --in-dir "$OUT" --title "$TITLE"

echo "Done: ${OUT}"
