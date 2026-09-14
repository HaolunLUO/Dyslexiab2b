#!/bin/bash
#SBATCH -J b2b_v4_basis
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/b2b_v4_basis_%j.out
#SBATCH -e logs/b2b_v4_basis_%j.err
#
# Frozen feature-basis preparation (Large V2, independent PCA, K=8).
# Submit from joint_v4/:  mkdir -p logs && sbatch scripts/run_prepare_basis.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export B2B_PCA_K="${B2B_PCA_K:-8}"
export B2B_BASIS_MODE="${B2B_BASIS_MODE:-independent}"
export B2B_EPOCH_ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
export B2B_FEAT_ACOUSTIC="${B2B_FEAT_ACOUSTIC:-whisperlargev2_acoustic}"
export B2B_FEAT_SPEECH="${B2B_FEAT_SPEECH:-whisperlargev2_speech}"
export B2B_FEAT_LANGUAGE="${B2B_FEAT_LANGUAGE:-whisperlargev2_language_audio_fused}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_joint_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
export B2B_NO_PLOT=1

module load julia/1.12.6

echo "Preparing basis → ${B2B_BASIS_DIR}  mode=${B2B_BASIS_MODE} K=${B2B_PCA_K}"
julia --project=.. prepare_feature_basis.jl
