#!/bin/bash
#SBATCH -J b2b_v4_splang_acr_off
#SBATCH -p mit_normal
#SBATCH -a 1-63%20
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/b2b_v4_splang_acr_off_%A_%a.out
#SBATCH -e logs/b2b_v4_splang_acr_off_%A_%a.err
#
# Observed B2B: Large V2 speech+language, offset-locked, acoustic-residual EEG.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=observed
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_offset_speechlang_acoures_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_PCA_K="${B2B_PCA_K:-8}"
export B2B_BASIS_MODE="${B2B_BASIS_MODE:-independent}"
export B2B_EPOCH_ANCHOR=offset
export B2B_FEATURE_SETS=whisperlargev2_speech,whisperlargev2_language_audio_fused
export B2B_FAMILY_NAMES=speech,language
export B2B_NO_PLOT=1

module load julia/1.12.6

echo "Offset speech+lang acoures task ${SLURM_ARRAY_TASK_ID}  out=${B2B_OUTDIR}"
julia --project=.. b2b_joint_v4_pipeline.jl "${SLURM_ARRAY_TASK_ID}"
