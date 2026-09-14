#!/bin/bash
#SBATCH -J b2b_hdc_dss40tb
#SBATCH -p mit_normal
#SBATCH -a 1-63%16
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 8:00:00
#SBATCH -o logs/b2b_hdc_dss40tb_%A_%a.out
#SBATCH -e logs/b2b_hdc_dss40tb_%A_%a.err
#
# HDC syntax acoures + DSS(40) train_half fit + 20 ms bin (leakage control).
# Fingerprint differs from dss40_bin20 via B2B_DSS_FIT_SCOPE=train_half.
#
#   sbatch --export=NONE scripts/run_hdc_snr_dss_trainhalf_observed_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

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
export B2B_DO_TG="${B2B_DO_TG:-0}"
export B2B_SPATIAL_DENOISE="${B2B_SPATIAL_DENOISE:-dss}"
export B2B_SPATIAL_N_COMP="${B2B_SPATIAL_N_COMP:-40}"
export B2B_TIME_BIN_MS="${B2B_TIME_BIN_MS:-20}"
export B2B_DSS_FIT_SCOPE=train_half
export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntactic_operation,hdc_syntactic_state,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40tb_bin20}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"

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

echo "HDC DSS train_half task ${SLURM_ARRAY_TASK_ID} (${PART}) out=${B2B_OUTDIR}"
julia --project=.. b2b_joint_v4_pipeline.jl "${SLURM_ARRAY_TASK_ID}"
