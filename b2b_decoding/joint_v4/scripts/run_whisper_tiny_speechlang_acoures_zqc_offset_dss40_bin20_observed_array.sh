#!/bin/bash
#SBATCH -J b2b_tiny_off_dss
#SBATCH -p mit_normal
#SBATCH -a 1-63%16
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/b2b_tiny_off_dss_%A_%a.out
#SBATCH -e logs/b2b_tiny_off_dss_%A_%a.err
#
# Observed: Tiny speech + language, offset, acoures zQC, DSS40 + 20 ms bin.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=observed
export B2B_EXTRACTOR_DIR="/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual"
export B2B_OUTDIR="/home/haolun52/orcd/pool/encoding_results_b2b_whispertiny_offset_speechlang_acoures_zqc_k8_dss40_bin20"
export B2B_BASIS_DIR="${B2B_OUTDIR}/_basis"
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR=offset
export B2B_FEATURE_SETS=whispertiny_speech,whispertiny_language_audio_fused
export B2B_FAMILY_NAMES=speech,language
export B2B_SPATIAL_DENOISE=dss
export B2B_SPATIAL_N_COMP=40
export B2B_TIME_BIN_MS=20
export B2B_TG_STRIDE=2
export B2B_DO_TG="${B2B_DO_TG:-0}"
export B2B_NO_PLOT=1

module load julia/1.12.6

PART=$(python3 - <<'PY'
import csv, os
cohort = os.path.join(os.environ.get("SLURM_SUBMIT_DIR", "."), "cohort_groups.csv")
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
rows = [r for r in csv.DictReader(open(cohort)) if r["include_primary"] in ("1", "True", "true")]
print(rows[idx - 1]["participant"])
PY
)
OUT_CSV="${B2B_OUTDIR}/${PART}/${PART}_b2b_family_agg.csv"
if [[ -f "$OUT_CSV" && "${B2B_FORCE:-0}" != "1" ]]; then
  echo "Skip ${PART}: ${OUT_CSV} exists"
  exit 0
fi

echo "Tiny speech+lang offset DSS+bin task ${SLURM_ARRAY_TASK_ID} (${PART}) out=${B2B_OUTDIR}"
test -f "${B2B_BASIS_DIR}/feature_basis.npz" || { echo "Missing basis"; exit 1; }
julia --project=.. b2b_joint_v4_pipeline.jl "${SLURM_ARRAY_TASK_ID}"
