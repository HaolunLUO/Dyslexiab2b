#!/bin/bash
#SBATCH -J b2b_split
#SBATCH -p mit_normal
#SBATCH -a 1-63%20
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/b2b_split_%A_%a.out
#SBATCH -e logs/b2b_split_%A_%a.err
#
# Phase 2: two observed B2B fits (odd/even 60 s) per subject.
# Submit only via submit_split_half_{onset,offset}.sh.

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
export B2B_NO_PLOT=1

: "${B2B_OUTDIR:?}"
: "${B2B_OBSERVED_OUTDIR:?}"
if [[ "$B2B_OUTDIR" == "$B2B_OBSERVED_OUTDIR" ]]; then
  echo "Refusing to write split-half into the observed outdir"; exit 1
fi
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OBSERVED_OUTDIR}/_basis}"

TID="${SLURM_ARRAY_TASK_ID:?}"
PART=$(python3 - <<PY
import csv, os
cohort = os.path.join(os.environ.get("SLURM_SUBMIT_DIR", "."), "cohort_groups.csv")
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
rows = [r for r in csv.DictReader(open(cohort)) if r["include_primary"] in ("1", "True", "true")]
print(rows[idx - 1]["participant"])
PY
)
DONE="${B2B_OUTDIR}/${PART}/${PART}_splithalf_scores.csv"
if [[ -f "$DONE" && "${B2B_FORCE:-0}" != "1" ]]; then
  echo "Skip ${PART}: $DONE exists"
  exit 0
fi

module load julia/1.12.6
echo "Split-half task $TID  $PART  out=${B2B_OUTDIR}"
julia --project=.. run_split_half.jl "$TID"
