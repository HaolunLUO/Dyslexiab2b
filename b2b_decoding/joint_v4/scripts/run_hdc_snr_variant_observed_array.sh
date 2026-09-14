#!/bin/bash
#SBATCH -J b2b_hdc_var
#SBATCH -p mit_normal
#SBATCH -a 1-63%16
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o logs/b2b_hdc_var_%A_%a.out
#SBATCH -e logs/b2b_hdc_var_%A_%a.err
#
# Generic HDC SNR-variant observed array. Required at submit time via --export:
#   B2B_OUTDIR, B2B_SPATIAL_DENOISE, B2B_TIME_BIN_MS
# Optional:
#   B2B_SPATIAL_N_COMP (40), B2B_LAMBDA_TUNE_ALL (0), B2B_FEATURE_SETS, …
#
# Example:
#   sbatch --export=ALL,B2B_OUTDIR=...,B2B_SPATIAL_DENOISE=dss,B2B_TIME_BIN_MS=0 \
#     scripts/run_hdc_snr_variant_observed_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

: "${B2B_OUTDIR:?B2B_OUTDIR must be set at sbatch --export}"
: "${B2B_SPATIAL_DENOISE:?B2B_SPATIAL_DENOISE must be set}"
: "${B2B_TIME_BIN_MS:?B2B_TIME_BIN_MS must be set}"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=observed
export B2B_EPOCH_ANCHOR=offset
export B2B_BASIS_MODE=independent
export B2B_PASSTHROUGH=1
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_NO_PLOT=1
export B2B_DO_TG="${B2B_DO_TG:-1}"
export B2B_TG_STRIDE="${B2B_TG_STRIDE:-2}"
export B2B_SPATIAL_N_COMP="${B2B_SPATIAL_N_COMP:-40}"
export B2B_LAMBDA_TUNE_ALL="${B2B_LAMBDA_TUNE_ALL:-0}"
export B2B_STRICT_FEAT_DIMS="${B2B_STRICT_FEAT_DIMS:-1}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
export B2B_FEATURE_SETS="${B2B_FEATURE_SETS:-hdc_phonetic:hdc_word_form:hdc_lexical_syntactic:hdc_syntactic_operation:hdc_syntactic_state:hdc_semantic}"
export B2B_FAMILY_NAMES="${B2B_FAMILY_NAMES:-phonetic:word_form:lexical_syntactic:syntactic_operation:syntactic_state:semantic}"

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
  if [[ "${B2B_DO_TG}" == "1" && ! -f "${B2B_OUTDIR}/${PART}/${PART}_tg_metrics.csv" ]]; then
    echo "Observed exists for ${PART}; TG missing → continue"
  else
    echo "Skip ${PART}: ${OUT_CSV} exists"
    exit 0
  fi
fi

echo "HDC variant task ${SLURM_ARRAY_TASK_ID} (${PART})"
echo "  out=${B2B_OUTDIR} spatial=${B2B_SPATIAL_DENOISE} bin=${B2B_TIME_BIN_MS} ltall=${B2B_LAMBDA_TUNE_ALL} TG=${B2B_DO_TG}"
echo "  features=${B2B_FEATURE_SETS}"
echo "  families=${B2B_FAMILY_NAMES}"
nfeat=$(python3 - <<'PY'
import os,re
print(len([x for x in re.split(r'[,;:]', os.environ.get('B2B_FEATURE_SETS','')) if x.strip()]))
PY
)
if [[ "$nfeat" -lt 2 ]]; then
  echo "ERROR: B2B_FEATURE_SETS parsed to $nfeat entries (sbatch --export comma bug?). Got: ${B2B_FEATURE_SETS}"
  exit 1
fi
test -f "${B2B_BASIS_DIR}/feature_basis.npz" || {
  echo "Missing basis at ${B2B_BASIS_DIR}; run prepare first"; exit 1;
}
julia --project=.. b2b_joint_v4_pipeline.jl "${SLURM_ARRAY_TASK_ID}"
