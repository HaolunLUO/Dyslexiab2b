#!/usr/bin/env bash
# Submit Tiny speech + GPT2-CN l24 on resegmented banded EEG.
# QOS allows ~8 jobs: first 7 arrays, then a chain for the rest + compare.
#
# Prerequisite: onset and offset feature bases (prepare_feature_basis).
# Usage (from joint_v4/):
#   bash scripts/submit_whisper_tiny_gpt2cn_banded_reseg.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
MAX_CONCUR="${MAX_CONCUR:-8}"
ARRAY="1-63%${MAX_CONCUR}"
# Do not inherit login-shell B2B paths from an earlier prepare step.
unset B2B_OUTDIR B2B_BASIS_DIR B2B_EXTRACTOR_DIR B2B_MODE B2B_OUT_SUFFIX || true

bash scripts/wire_banded_b2b_extractors.sh

for anchor in onset offset; do
  test -f "${POOL}/encoding_results_b2b_tinyspeech_gpt2cn_${anchor}_k8/_basis/feature_basis.npz" || {
    echo "Missing ${anchor} basis. Run: sbatch --export=ALL,B2B_EPOCH_ANCHOR=${anchor} scripts/run_whisper_tiny_gpt2cn_prepare_basis.sh"
    exit 1
  }
done

WAVE1=(
  "delta onset"
  "delta offset"
  "theta onset"
  "theta offset"
  "broadband onset"
  "broadband offset"
  "alpha onset"
)

ids=()
for spec in "${WAVE1[@]}"; do
  set -- $spec
  band=$1; anchor=$2
  jid=$(sbatch --parsable \
    --job-name="tgpt2_${band}_${anchor}" \
    --array="$ARRAY" \
    --export="B2B_BAND=${band},B2B_EPOCH_ANCHOR=${anchor},JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia},POOL=${POOL}" \
    scripts/run_whisper_tiny_gpt2cn_banded_observed_array.sh)
  echo "submitted ${band}/${anchor} job=${jid}"
  ids+=("$jid")
done

dep=$(IFS=:; echo "${ids[*]}")
chain=$(sbatch --parsable \
  --job-name="tgpt2_chain" \
  --dependency="afterany:${dep}" \
  --partition=mit_normal --cpus-per-task=1 --mem=2G --time=00:20:00 \
  --output=logs/b2b_tiny_gpt2cn_chain_%j.out \
  --error=logs/b2b_tiny_gpt2cn_chain_%j.err \
  scripts/submit_whisper_tiny_gpt2cn_banded_wave2.sh)
echo "chain job=${chain} after ${dep}"
echo "${ids[*]} ${chain}" > logs/tiny_gpt2cn_reseg_slurm_ids.txt
