#!/bin/bash
#SBATCH -J b2b_exist
#SBATCH -p mit_normal
#SBATCH -a 1-63%20
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/b2b_exist_%A_%a.out
#SBATCH -e logs/b2b_exist_%A_%a.err
#
# Phase 1 existence nulls: 63 subjects, 200 cached circular shifts.
# Submit only via submit_existence_nulls_{onset,offset}.sh (hard-sets outdirs).

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

: "${B2B_OUTDIR:?B2B_OUTDIR must be the existence outdir}"
: "${B2B_OBSERVED_OUTDIR:?B2B_OBSERVED_OUTDIR must be the frozen observed outdir}"
if [[ "$B2B_OUTDIR" == "$B2B_OBSERVED_OUTDIR" ]]; then
  echo "Refusing to write existence nulls into the observed outdir"; exit 1
fi
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OBSERVED_OUTDIR}/_basis}"
export B2B_EXISTENCE_MIN_SHIFT="${B2B_EXISTENCE_MIN_SHIFT:-20}"
export B2B_EXISTENCE_N_NULL="${B2B_EXISTENCE_N_NULL:-200}"
export B2B_N_NULL="${B2B_EXISTENCE_N_NULL}"
export B2B_NO_PLOT=1

TID="${SLURM_ARRAY_TASK_ID:?}"
PART=$(python3 - <<PY
import csv, os
cohort = os.path.join(os.environ.get("SLURM_SUBMIT_DIR", "."), "cohort_groups.csv")
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
rows = [r for r in csv.DictReader(open(cohort)) if r["include_primary"] in ("1", "True", "true")]
print(rows[idx - 1]["participant"])
PY
)
DONE="${B2B_OUTDIR}/${PART}/${PART}_existence_confirm.csv"
if [[ -f "$DONE" && "${B2B_FORCE:-0}" != "1" ]]; then
  echo "Skip ${PART}: $DONE exists"
  exit 0
fi

module load julia/1.12.6
echo "Existence null task $TID  $PART  obs=${B2B_OBSERVED_OUTDIR}  out=${B2B_OUTDIR}"
julia --project=.. run_existence_nulls.jl "$TID"
