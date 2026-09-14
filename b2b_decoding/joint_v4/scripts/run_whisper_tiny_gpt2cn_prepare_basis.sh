#!/bin/bash
#SBATCH -J b2b_tiny_gpt2cn_basis
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o logs/b2b_tiny_gpt2cn_basis_%j.out
#SBATCH -e logs/b2b_tiny_gpt2cn_basis_%j.err
#
# Feature basis: Whisper Tiny speech + GPT2-CN l24 (EEG-independent PCA).
# Required env: B2B_EPOCH_ANCHOR=onset|offset

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

ANCHOR="${B2B_EPOCH_ANCHOR:?set B2B_EPOCH_ANCHOR}"
POOL="${POOL:-/orcd/pool/005/haolun52}"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_shared}"
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR="$ANCHOR"
export B2B_FEATURE_SETS=whispertiny_speech,gpt2_l24
export B2B_FAMILY_NAMES=speech,gpt2cn
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_NO_PLOT=1
export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_tinyspeech_gpt2cn_${ANCHOR}_k8}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"

module load julia/1.12.6
echo "Preparing Tiny-speech + GPT2-CN l24 basis (${ANCHOR}) → ${B2B_BASIS_DIR}"
test -f "${B2B_EXTRACTOR_DIR}/_shared_wordlocked_features/section_001/X_word_gpt2_l24.npy"
test -f "${B2B_EXTRACTOR_DIR}/_shared_wordlocked_features/section_001/X_word_whispertiny_speech.npy"
julia --project=.. prepare_feature_basis.jl
