#!/bin/bash
#SBATCH -J b2b_tiny_splang_dss
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o logs/b2b_tiny_splang_dss_basis_%j.out
#SBATCH -e logs/b2b_tiny_splang_dss_basis_%j.err
#
# Whisper Tiny speech + language, onset, acoures zQC, DSS40 + 20 ms bin.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export B2B_EXTRACTOR_DIR="/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual"
export B2B_OUTDIR="/home/haolun52/orcd/pool/encoding_results_b2b_whispertiny_onset_speechlang_acoures_zqc_k8_dss40_bin20"
export B2B_BASIS_DIR="${B2B_OUTDIR}/_basis"
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR=onset
export B2B_FEATURE_SETS=whispertiny_speech,whispertiny_language_audio_fused
export B2B_FAMILY_NAMES=speech,language
export B2B_SPATIAL_DENOISE=dss
export B2B_SPATIAL_N_COMP=40
export B2B_TIME_BIN_MS=20
export B2B_TG_STRIDE=2
export B2B_NO_PLOT=1

module load julia/1.12.6
mkdir -p "${B2B_OUTDIR}" "${B2B_BASIS_DIR}"
rm -f "${B2B_OUTDIR}/_analysis_manifest.json"
echo "Preparing Tiny speech+lang onset DSS+bin basis → ${B2B_BASIS_DIR}"
julia --project=.. prepare_feature_basis.jl
