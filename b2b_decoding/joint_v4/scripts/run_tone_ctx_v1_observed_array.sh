#!/bin/bash
#SBATCH -J b2b_tctx
#SBATCH -p mit_normal
#SBATCH -a 1-63%16
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH --exclude=node1617
#SBATCH -o logs/b2b_tctx_%A_%a.out
#SBATCH -e logs/b2b_tctx_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
POOL="${POOL:-/orcd/pool/005/haolun52}"
SHARED_DEPOT="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
LOCAL_COMPILED="/tmp/${USER}-julia-compiled"
case ":${SHARED_DEPOT}:" in
  *:${LOCAL_COMPILED}:*) export JULIA_DEPOT_PATH="${SHARED_DEPOT}" ;;
  *) export JULIA_DEPOT_PATH="${LOCAL_COMPILED}:${SHARED_DEPOT}" ;;
esac
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic}"
mkdir -p "${LOCAL_COMPILED}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=observed
TID="${SLURM_ARRAY_TASK_ID:?}"
if [[ -n "${B2B_PILOT_SUBJECTS:-}" ]]; then
  PART=$(python3 - <<PY
import os
subs = [s for s in os.environ["B2B_PILOT_SUBJECTS"].split(",") if s.strip()]
idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
print(subs[idx])
PY
)
else
  PART="$TID"
fi
module load julia/1.12.6
echo "tone_ctx_v1 observed $PART  feats=${B2B_FEATURE_SETS}  out=${B2B_OUTDIR}"
julia --project=.. b2b_joint_v4_pipeline.jl "$PART"
