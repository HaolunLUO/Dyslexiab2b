#!/bin/bash
#SBATCH -J b2b_v4_sens
#SBATCH -p mit_normal
#SBATCH -a 1-252%20
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/b2b_v4_sens_%A_%a.out
#SBATCH -e logs/b2b_v4_sens_%A_%a.err
#
# Observed-only sensitivities (after locking primary):
#   1-63:   K=4  independent onset
#   64-126: K=16 independent onset
#   127-189: text-only language (K=8)
#   190-252: ordered basis (K=8)
#
# Offset locking is a separate outdir / job (see README).
#
# Prep matching bases first (each outdir gets its own _basis).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export B2B_MODE=observed
export B2B_NO_PLOT=1
export B2B_EPOCH_ANCHOR=onset
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
export B2B_FEAT_ACOUSTIC=whisperlargev2_acoustic
export B2B_FEAT_SPEECH=whisperlargev2_speech

TID="${SLURM_ARRAY_TASK_ID:?}"
if (( TID <= 63 )); then
  export B2B_PCA_K=4
  export B2B_BASIS_MODE=independent
  export B2B_FEAT_LANGUAGE=whisperlargev2_language_audio_fused
  PART_IDX=$TID
  export B2B_OUTDIR="${B2B_OUTDIR_K4:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_joint_k4}"
elif (( TID <= 126 )); then
  export B2B_PCA_K=16
  export B2B_BASIS_MODE=independent
  export B2B_FEAT_LANGUAGE=whisperlargev2_language_audio_fused
  PART_IDX=$(( TID - 63 ))
  export B2B_OUTDIR="${B2B_OUTDIR_K16:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_joint_k16}"
elif (( TID <= 189 )); then
  export B2B_PCA_K=8
  export B2B_BASIS_MODE=independent
  export B2B_FEAT_LANGUAGE=whisperlargev2_language_text_only
  PART_IDX=$(( TID - 126 ))
  export B2B_OUTDIR="${B2B_OUTDIR_TEXT:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_textonly_k8}"
else
  export B2B_PCA_K=8
  export B2B_BASIS_MODE=ordered
  export B2B_FEAT_LANGUAGE=whisperlargev2_language_audio_fused
  PART_IDX=$(( TID - 189 ))
  export B2B_OUTDIR="${B2B_OUTDIR_ORD:-/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_ordered_k8}"
fi
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"

module load julia/1.12.6
echo "Sensitivity K=$B2B_PCA_K mode=$B2B_BASIS_MODE lang=$B2B_FEAT_LANGUAGE idx=$PART_IDX → $B2B_OUTDIR"
julia --project=.. b2b_joint_v4_pipeline.jl "$PART_IDX"
