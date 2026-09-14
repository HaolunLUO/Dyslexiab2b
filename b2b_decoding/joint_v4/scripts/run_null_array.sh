#!/bin/bash
#SBATCH -J b2b_v4_null
#SBATCH -p mit_normal
#SBATCH -a 1-315%24
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 2:00:00
#SBATCH -o logs/b2b_v4_null_%A_%a.out
#SBATCH -e logs/b2b_v4_null_%A_%a.err
#
# Null chunks: 63 participants × 5 chunks of 40 reps = 315 array tasks.
# Array id → participant = ((id-1) % 63) + 1
#            chunk       = ((id-1) / 63)     → null_start = chunk*40 + 1
#
# Resource notes (mit_normal / QOSMaxCpuPerUserLimit ≈ 96 CPUs):
#   -c 4 → max useful concurrency ≈ 24 (96/4); %40 only created pending noise.
#   --mem=4G: measured MaxRSS ≈ 2 GB; 16G over-request hurt packing.
#   -t 2:00:00: chunk wall ≈ 30–80 min; 12h inflated queue priority cost.
# Override via sbatch CLI if needed: --array=1-315%24 --mem=4G -t 2:00:00
#
# Benchmark one chunk first:
#   sbatch --array=1 scripts/run_null_array.sh
#
# Full nulls (only after observed stop-gates pass):
#   sbatch scripts/run_null_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=null
export B2B_PCA_K="${B2B_PCA_K:-8}"
export B2B_BASIS_MODE="${B2B_BASIS_MODE:-independent}"
export B2B_EPOCH_ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
export B2B_FEAT_ACOUSTIC="${B2B_FEAT_ACOUSTIC:-whisperlargev2_acoustic}"
export B2B_FEAT_SPEECH="${B2B_FEAT_SPEECH:-whisperlargev2_speech}"
export B2B_FEAT_LANGUAGE="${B2B_FEAT_LANGUAGE:-whisperlargev2_language_audio_fused}"
export B2B_N_NULL=200
export B2B_NULL_COUNT=40
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_joint_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
export B2B_NO_PLOT=1

TID="${SLURM_ARRAY_TASK_ID:?}"
PART_IDX=$(( (TID - 1) % 63 + 1 ))
CHUNK=$(( (TID - 1) / 63 ))
export B2B_NULL_START=$(( CHUNK * 40 + 1 ))
export PART_IDX

# Resolve participant for skip / logging (idempotent requeue).
PART=$(python3 - <<'PY'
import csv, os
cohort = os.path.join(os.environ.get("SLURM_SUBMIT_DIR", "."), "cohort_groups.csv")
idx = int(os.environ["PART_IDX"])
rows = [r for r in csv.DictReader(open(cohort)) if r["include_primary"] in ("1", "True", "true")]
print(rows[idx - 1]["participant"])
PY
)

CHUNK_END=$(( B2B_NULL_START + B2B_NULL_COUNT - 1 ))
CHUNK_TAG=$(printf "%04d_%04d" "$B2B_NULL_START" "$CHUNK_END")
CHUNK_CSV="${B2B_OUTDIR}/${PART}/${PART}_null_${CHUNK_TAG}_traces.csv"
AGG_CSV="${B2B_OUTDIR}/${PART}/${PART}_null_traces.csv"
if [[ -f "$AGG_CSV" && "${B2B_FORCE:-0}" != "1" ]]; then
  echo "Skip ${PART}: aggregated null_traces already exist"
  exit 0
fi
if [[ -f "$CHUNK_CSV" && "${B2B_FORCE:-0}" != "1" ]]; then
  echo "Skip ${PART} chunk ${CHUNK_TAG}: ${CHUNK_CSV} exists"
  exit 0
fi

module load julia/1.12.6

echo "Null task $TID  ${PART} (idx=$PART_IDX)  null_start=$B2B_NULL_START count=$B2B_NULL_COUNT  out=${B2B_OUTDIR}"
julia --project=.. b2b_joint_v4_pipeline.jl "$PART_IDX"
