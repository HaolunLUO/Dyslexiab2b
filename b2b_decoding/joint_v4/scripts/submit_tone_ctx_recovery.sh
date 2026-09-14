#!/usr/bin/env bash
# Recover the interrupted frozen M1/M2/M3 adjudication (2026-09-08).
set -euo pipefail
cd "$(dirname "$0")/.."
export POOL=/orcd/pool/005/haolun52
export JULIA_DEPOT_PATH="${POOL}/.julia"
export JULIA_CPU_TARGET=generic
mkdir -p logs
for MODEL in M1 M2 M3; do
  export MODEL
  source scripts/tone_ctx_v1_env.sh
  export B2B_OBSERVED_OUTDIR="$B2B_OUTDIR"
  test -f "$B2B_BASIS_DIR/feature_basis.npz"
  export B2B_OUTDIR="${B2B_OBSERVED_OUTDIR}_existence_nulls"
  export B2B_EXISTENCE_MIN_SHIFT=20 B2B_EXISTENCE_N_NULL=5000 B2B_N_NULL=5000
  export B2B_EXISTENCE_NULL=freedman_lane
  ARRAY=$(python3 - <<'PYC'
import csv, os
from pathlib import Path
parts = [r['participant'] for r in csv.DictReader(open('cohort_groups.csv')) if r['include_primary'] in ('1','True','true')]
root = Path(os.environ['B2B_OUTDIR'])
print(','.join(str(i) for i,p in enumerate(parts,1) if not (root/p/f'{p}_existence_confirm.csv').is_file()))
PYC
)
  if [[ -n "$ARRAY" ]]; then
    job=$(sbatch --parsable --job-name="tctxe_${MODEL}_recover" --partition=mit_normal --exclude=node1617 --array="${ARRAY}%20" --time=8:00:00 --mem=16G --export=ALL scripts/run_existence_null_array.sh)
    echo "$MODEL existence $job" | tee -a logs/tone_ctx_recovery_20260908_jobs.txt
  fi
  export B2B_OUTDIR="${B2B_OBSERVED_OUTDIR}_splithalf"
  job=$(sbatch --parsable --job-name="tctxs_${MODEL}_recover" --partition=mit_normal --exclude=node1617 --array=1-63%20 --time=4:00:00 --mem=16G --export=ALL scripts/run_split_half_array.sh)
  echo "$MODEL splithalf $job" | tee -a logs/tone_ctx_recovery_20260908_jobs.txt
done
