#!/usr/bin/env bash
# After wave1 Tiny-speech+GPT2CN arrays finish, submit remaining 3 arms + compare.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAX_CONCUR="${MAX_CONCUR:-8}"
ARRAY="1-63%${MAX_CONCUR}"

ids=()
for spec in "alpha offset" "beta onset" "beta offset"; do
  set -- $spec
  band=$1; anchor=$2
  jid=$(sbatch --parsable \
    --job-name="tgpt2_${band}_${anchor}" \
    --array="$ARRAY" \
    --export="B2B_BAND=${band},B2B_EPOCH_ANCHOR=${anchor},JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia},POOL=${POOL:-/orcd/pool/005/haolun52}" \
    scripts/run_whisper_tiny_gpt2cn_banded_observed_array.sh)
  echo "chained ${band}/${anchor} job=${jid}"
  ids+=("$jid")
done

dep=$(IFS=:; echo "${ids[*]}")
cmp=$(sbatch --parsable \
  --job-name="tgpt2_compare" \
  --dependency="afterany:${dep}" \
  --partition=mit_normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=02:00:00 \
  --output=logs/b2b_tiny_gpt2cn_compare_%j.out \
  --error=logs/b2b_tiny_gpt2cn_compare_%j.err \
  --wrap="export SCORE_COLS=mean_trace; bash scripts/compare_whisper_tiny_gpt2cn_banded.sh")
echo "submitted compare job=${cmp} after ${dep}"
echo "${ids[*]} ${cmp}" >> logs/tiny_gpt2cn_reseg_slurm_ids.txt
