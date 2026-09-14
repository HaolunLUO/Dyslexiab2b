#!/usr/bin/env bash
# Submit 10 Tiny speech+lang banded arrays (5 bands × onset/offset) to mit_normal.
# Usage (from joint_v4/):
#   bash scripts/submit_whisper_tiny_speechlang_banded_reseg.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export B2B_OUT_SUFFIX="${B2B_OUT_SUFFIX:-_reseg}"
export B2B_FORCE="${B2B_FORCE:-0}"
MAX_CONCUR="${MAX_CONCUR:-8}"

bash scripts/wire_banded_b2b_extractors.sh

BANDS="${BANDS:-delta theta broadband alpha beta}"
ANCHORS="${ANCHORS:-onset offset}"
ids=()

for band in $BANDS; do
  for anchor in $ANCHORS; do
    jid=$(sbatch --parsable \
      --job-name="tiny_${band}_${anchor}" \
      --array="1-63%${MAX_CONCUR}" \
      --export="ALL,B2B_BAND=${band},B2B_EPOCH_ANCHOR=${anchor},B2B_OUT_SUFFIX=${B2B_OUT_SUFFIX},B2B_FORCE=${B2B_FORCE}" \
      scripts/run_whisper_tiny_speechlang_banded_observed_array.sh)
    echo "submitted ${band}/${anchor} job=${jid}"
    ids+=("$jid")
  done
done

dep=$(IFS=:; echo "${ids[*]}")
cmp=$(sbatch --parsable \
  --job-name="tiny_banded_compare" \
  --dependency="afterany:${dep}" \
  --partition=mit_normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=02:00:00 \
  --output=logs/b2b_tiny_banded_compare_%j.out \
  --error=logs/b2b_tiny_banded_compare_%j.err \
  --wrap="export B2B_OUT_SUFFIX=${B2B_OUT_SUFFIX}; export SCORE_COLS=mean_trace; bash scripts/compare_whisper_tiny_speechlang_banded.sh")
echo "submitted compare job=${cmp} after ${dep}"
echo "${ids[*]} ${cmp}" > logs/banded_tiny_reseg_slurm_ids.txt
