#!/bin/bash
#SBATCH -J b2b_mfa_v2
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -a 1-63%20
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o logs/b2b_mfa_v2_%A_%a.out
#SBATCH -e logs/b2b_mfa_v2_%A_%a.err
#
# Full cohort: onset-locked B2B on envelope_v2 + MFA phones_v2 + tone_v2
# (passthrough, OLS H). Provenance-aware skip.
# Submit: mkdir -p logs && sbatch scripts/run_mfa_phone_tone_env_observed_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=observed
export B2B_EPOCH_ANCHOR=onset
export B2B_BASIS_MODE=independent
export B2B_PASSTHROUGH=1
export B2B_H_RIDGE_KAPPA=0
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_STRICT_FEAT_DIMS=0
export B2B_NO_PLOT=1
export B2B_FEATURE_SETS=envelope_v2,mfa_phones_v2,mfa_tone_v2
export B2B_FAMILY_NAMES=envelope,phones,tone
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_mfa_v2_phone_tone_env_onset_passthrough}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"

# shellcheck source=mfa_v2_provenance.sh
source scripts/mfa_v2_provenance.sh

module load julia/1.12.6

echo "MFA v2 phone/tone/env observed task ${SLURM_ARRAY_TASK_ID}  out=${B2B_OUTDIR}"
p=$(awk -F, -v i="${SLURM_ARRAY_TASK_ID}" 'NR>1 && $3==1 {n++; if(n==i) {print $1; exit}}' cohort_groups.csv)
if [[ -z "${p:-}" ]]; then
  echo "ERROR: could not resolve participant for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

if mfa_v2_should_skip_subject "$p"; then
  exit 0
fi

julia --project=.. b2b_joint_v4_pipeline.jl "${SLURM_ARRAY_TASK_ID}"
