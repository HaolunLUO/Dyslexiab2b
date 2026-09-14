#!/bin/bash
#SBATCH -J b2b_v4_sptglv_t05_basis
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/b2b_v4_sptglv_t05_basis_%j.out
#SBATCH -e logs/b2b_v4_sptglv_t05_basis_%j.err
#
# Basis: Tiny speech + GPT2-CN l24 + GloVe, onset, acoustic-residual EEG,
# extended pre-onset (−0.5 … 1.0 s).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_tiny_speech_gpt2_glove_onset_acoures_tmin-0.5_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_PCA_K="${B2B_PCA_K:-8}"
export B2B_BASIS_MODE="${B2B_BASIS_MODE:-independent}"
export B2B_EPOCH_ANCHOR=onset
export B2B_TMIN_S=-0.5
export B2B_TMAX_S=1.0
export B2B_FEATURE_SETS=whispertiny_speech,gpt2_l24,glove
export B2B_FAMILY_NAMES=speech,gpt2cn,glove
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_NO_PLOT=1

module load julia/1.12.6

echo "Preparing Tiny-speech/GPT2/GloVe basis (tmin=${B2B_TMIN_S}) → ${B2B_BASIS_DIR}"
julia --project=.. prepare_feature_basis.jl
