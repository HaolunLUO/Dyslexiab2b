#!/bin/bash
#SBATCH -J n400_rev_lmm
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/n400_rev_lmm_%j.out
#SBATCH -e logs/n400_rev_lmm_%j.err
# Core revision LMMs. 480 s is not a stopping rule; this wall is.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs
source /etc/profile.d/00-modulepath.sh
module load R/4.5.2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "host $(hostname) job ${SLURM_JOB_ID:-local} start $(date -Is)"
OUT="/orcd/pool/005/haolun52/n400_storytime_v1/revision/lmm"
set +e
Rscript fit_revision_lmm.R \
  /orcd/pool/005/haolun52/n400_storytime_v1/stimulus/lmm_input.csv \
  "${OUT}"
rc=$?
set -e
if [[ "${rc}" -eq 2 ]] || [[ -f "${OUT}/region_FAILED.txt" ]]; then
  echo "lme4 did not converge a required model; MixedModels fallback"
  module load julia/1.12.6
  export JULIA_DEPOT_PATH="/orcd/pool/005/haolun52/.julia"
  ENV="/orcd/pool/005/haolun52/n400_storytime_v1/revision/julia_lmm"
  mkdir -p "${ENV}"
  if [[ ! -f "${ENV}/Manifest.toml" ]]; then
    julia --project="${ENV}" -e 'using Pkg; Pkg.add(["MixedModels", "CSV", "DataFrames", "CategoricalArrays", "StatsModels"])'
  fi
  set +e
  julia --project="${ENV}" fit_mixedmodels.jl "${OUT}"
  mm=$?
  set -e
  if [[ "${mm}" -ne 0 ]]; then
    echo "NEED_BAYES MixedModels failed with ${mm}" | tee "${OUT}/NEED_BAYES.txt"
    echo "end $(date -Is) status ${mm}"
    exit "${mm}"
  fi
fi
echo "end $(date -Is) status ${rc}"
# Core-floor failure is recovered when MixedModels writes core_mixed_OK.txt.
if [[ "${rc}" -eq 2 && -f "${OUT}/core_mixed_OK.txt" ]]; then
  exit 0
fi
exit "${rc}"
