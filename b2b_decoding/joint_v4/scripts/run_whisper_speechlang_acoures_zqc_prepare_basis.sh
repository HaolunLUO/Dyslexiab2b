#!/bin/bash
#SBATCH -J b2b_v4_splang_zqc_basis
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/b2b_v4_splang_zqc_basis_%j.out
#SBATCH -e logs/b2b_v4_splang_zqc_basis_%j.err
#
# Large V2 speech + language, acoures EEG, zscore-only epoch QC (no µV PTP).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_speechlang_acoures_zqc_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR=onset
export B2B_FEATURE_SETS=whisperlargev2_speech,whisperlargev2_language_audio_fused
export B2B_FAMILY_NAMES=speech,language
export B2B_NO_PLOT=1

module load julia/1.12.6
echo "Preparing speech+language basis (acoures, zqc) → ${B2B_BASIS_DIR}"
julia --project=.. prepare_feature_basis.jl
