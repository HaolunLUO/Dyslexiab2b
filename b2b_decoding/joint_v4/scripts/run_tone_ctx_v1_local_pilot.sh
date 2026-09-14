#!/usr/bin/env bash
# Login-node fallback for the pre-registered 5-subject × M0–M3 smoke.
# Used only when slurmctld is down. Same env as submit_tone_ctx_v1.sh --pilot.
set -euo pipefail
cd "$(dirname "$0")/.."
POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic}"
export B2B_NO_PLOT=1
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-4}"
PILOT_SUBJECTS="${PILOT_SUBJECTS:-RN102,RN103,RN105,D001d,D011d}"
MODELS="${MODELS:-M0,M1,M2,M3}"
module load julia/1.12.6
IFS=',' read -r -a SUBS <<< "$PILOT_SUBJECTS"
IFS=',' read -r -a ARMS <<< "$MODELS"
n_run=0
n_skip=0
for MODEL in "${ARMS[@]}"; do
  export MODEL
  # shellcheck disable=SC1091
  source scripts/tone_ctx_v1_env.sh
  test -f "${B2B_BASIS_DIR}/feature_basis.npz" || {
    echo "Missing basis for $MODEL — run prepare_feature_basis.jl first"
    exit 1
  }
  for S in "${SUBS[@]}"; do
    qc="${B2B_OUTDIR}/${S}/${S}_qc_summary.json"
    if [[ -f "$qc" ]] && python3 - "$qc" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if int(d.get("n_valid_partitions", 0)) >= 1 and not d.get("any_invalid", True) else 1)
PY
    then
      echo "skip $MODEL $S (already valid)"
      n_skip=$((n_skip + 1))
      continue
    fi
    echo "===== local pilot $MODEL $S → ${B2B_OUTDIR} ====="
    julia --project=.. b2b_joint_v4_pipeline.jl "$S"
    n_run=$((n_run + 1))
  done
done
echo "local pilot done  ran=${n_run} skipped=${n_skip}"
