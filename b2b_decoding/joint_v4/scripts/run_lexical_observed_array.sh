#!/bin/bash
#SBATCH -J b2b_lex_obs
#SBATCH -p mit_normal
#SBATCH -a 1-63%20
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o logs/b2b_lex_obs_%A_%a.out
#SBATCH -e logs/b2b_lex_obs_%A_%a.err
#
# Lexical King-style B2B observed array (duration / frequency / GPT2CN surprisal).
# Submit: mkdir -p logs && sbatch scripts/run_lexical_observed_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=observed
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR=onset
export B2B_FEAT_ACOUSTIC=lexical_duration
export B2B_FEAT_SPEECH=lexical_frequency
export B2B_FEAT_LANGUAGE=gpt2cn_surprisal
export B2B_FAMILY_NAMES=duration,frequency,surprisal
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_lexical_onset_dur_freq_surp_k1}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
export B2B_NO_PLOT=1

module load julia/1.12.6

echo "Lexical observed task ${SLURM_ARRAY_TASK_ID}  out=${B2B_OUTDIR}"
julia --project=.. b2b_joint_v4_pipeline.jl "${SLURM_ARRAY_TASK_ID}"
