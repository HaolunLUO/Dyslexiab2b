#!/bin/bash
#SBATCH -J b2b_tiny_band
#SBATCH -p mit_normal
#SBATCH -a 1-63%16
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/b2b_tiny_band_%A_%a.out
#SBATCH -e logs/b2b_tiny_band_%A_%a.err
#
# Observed: Whisper Tiny speech+language on banded EEG (zQC, no DSS).
# Required env (via sbatch --export): B2B_BAND, B2B_EPOCH_ANCHOR
# Optional: B2B_OUTDIR, B2B_BASIS_DIR, B2B_EXTRACTOR_DIR
#
# Example:
#   bash scripts/wire_banded_b2b_extractors.sh
#   sbatch --export=ALL,B2B_BAND=delta,B2B_EPOCH_ANCHOR=onset \
#     scripts/run_whisper_tiny_speechlang_banded_observed_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

BAND="${B2B_BAND:?set B2B_BAND}"
ANCHOR="${B2B_EPOCH_ANCHOR:?set B2B_EPOCH_ANCHOR}"
POOL="${POOL:-/orcd/pool/005/haolun52}"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=observed
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR="$ANCHOR"
export B2B_FEATURE_SETS=whispertiny_speech,whispertiny_language_audio_fused
export B2B_FAMILY_NAMES=speech,language
export B2B_NO_PLOT=1
export B2B_DO_TG=0
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_banded_b2b/${BAND}}"
export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_whispertiny_${ANCHOR}_speechlang_${BAND}_zqc_k8${B2B_OUT_SUFFIX:-}}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${POOL}/encoding_results_b2b_whispertiny_${ANCHOR}_speechlang_acoures_zqc_k8/_basis}"

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

echo "Tiny speech+lang ${BAND} ${ANCHOR} task ${SLURM_ARRAY_TASK_ID} (${PART}) out=${B2B_OUTDIR}"
test -d "$B2B_EXTRACTOR_DIR" || { echo "Missing extractor $B2B_EXTRACTOR_DIR"; exit 1; }
test -f "${B2B_BASIS_DIR}/feature_basis.npz" || { echo "Missing basis"; exit 1; }
julia --project=.. b2b_joint_v4_pipeline.jl "${SLURM_ARRAY_TASK_ID}"
