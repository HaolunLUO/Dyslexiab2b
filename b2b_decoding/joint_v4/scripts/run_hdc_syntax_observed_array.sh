#!/bin/bash
#SBATCH -J b2b_hdc_syn
#SBATCH -p mit_normal
#SBATCH -a 1-63%16
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 8:00:00
#SBATCH -o logs/b2b_hdc_syn_%A_%a.out
#SBATCH -e logs/b2b_hdc_syn_%A_%a.err
#
# Tree-syntax HDC observed array (offset-locked).
# Submit: mkdir -p logs && sbatch scripts/run_hdc_syntax_observed_array.sh
# Optional: B2B_DO_TG=1 sbatch scripts/run_hdc_syntax_observed_array.sh

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
export B2B_DO_TG="${B2B_DO_TG:-1}"
export B2B_TG_STRIDE="${B2B_TG_STRIDE:-2}"
export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntactic_operation,hdc_syntactic_state,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"

module load julia/1.12.6

echo "HDC syntax observed task ${SLURM_ARRAY_TASK_ID}  out=${B2B_OUTDIR}  TG=${B2B_DO_TG}"
p=$(awk -F, -v i="${SLURM_ARRAY_TASK_ID}" 'NR>1 && $3==1 {n++; if(n==i) {print $1; exit}}' cohort_groups.csv)
agg="${B2B_OUTDIR}/${p}/${p}_b2b_family_agg.csv"
tg="${B2B_OUTDIR}/${p}/${p}_tg_metrics.csv"
if [[ -f "$agg" && "${B2B_FORCE:-0}" != "1" ]]; then
  if [[ "${B2B_DO_TG}" != "1" || -f "$tg" ]]; then
    echo "[skip] ${SLURM_ARRAY_TASK_ID} $p"
    exit 0
  fi
fi
julia --project=.. b2b_joint_v4_pipeline.jl "${SLURM_ARRAY_TASK_ID}"
