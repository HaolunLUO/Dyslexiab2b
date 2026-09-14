#!/bin/bash
#SBATCH -J b2b_ica_env
#SBATCH -p mit_normal
#SBATCH -a 1-63%20
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o logs/b2b_ica_envpitch_%A_%a.out
#SBATCH -e logs/b2b_ica_envpitch_%A_%a.err
#
# MNE-ICA v1 EEG × envelope_v2 / pitch / word_onset / frequency / GPT2CN surprisal.
# Passthrough families. Submit from joint_v4/:
#   bash scripts/submit_mne_ica_v1_envpitch_array.sh
#   B2B_EPOCH_ANCHOR=offset bash scripts/submit_mne_ica_v1_envpitch_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
# Prepend node-local compiled cache so mixed CPU types on mit_preemptable
# do not stall on "compatible target in cached code image".
SHARED_DEPOT="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
LOCAL_COMPILED="/tmp/${USER}-julia-compiled"
case ":${SHARED_DEPOT}:" in
  *:${LOCAL_COMPILED}:*) export JULIA_DEPOT_PATH="${SHARED_DEPOT}" ;;
  *) export JULIA_DEPOT_PATH="${LOCAL_COMPILED}:${SHARED_DEPOT}" ;;
esac
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic}"
mkdir -p "${LOCAL_COMPILED}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE="${B2B_MODE:-observed}"
export B2B_EPOCH_ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
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
export B2B_FEATURE_SETS="${B2B_FEATURE_SETS:-envelope_v2:pitch:word_onset:lexical_frequency:gpt2cn_surprisal}"
export B2B_FAMILY_NAMES="${B2B_FAMILY_NAMES:-envelope:pitch:onset:frequency:surprisal}"
export B2B_TMIN_S="${B2B_TMIN_S:--0.2}"
export B2B_TMAX_S="${B2B_TMAX_S:-1.0}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b}"
if [[ -z "${B2B_OUTDIR:-}" ]]; then
  if [[ "$B2B_EPOCH_ANCHOR" == "offset" ]]; then
    export B2B_OUTDIR="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_offset_freq_surp_passthrough"
  else
    export B2B_OUTDIR="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_onset_freq_surp_passthrough"
  fi
fi
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"

module load julia/1.12.6

echo "ICA env/pitch observed task ${SLURM_ARRAY_TASK_ID}  anchor=${B2B_EPOCH_ANCHOR}  tmin=${B2B_TMIN_S} tmax=${B2B_TMAX_S}  feats=${B2B_FEATURE_SETS}  out=${B2B_OUTDIR}"
julia --project=.. b2b_joint_v4_pipeline.jl "${SLURM_ARRAY_TASK_ID}"
