#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=logs/extract_whisper_encoder_%j.out
#SBATCH --error=logs/extract_whisper_encoder_%j.err
#
# Submit a GPU job to re-extract Whisper encoder (acoustic + speech) word-locked
# embeddings from lPPCN section WAVs.
#
# Usage example:
#   sbatch --export=ALL,SECTION_ID=1,MODEL_NAME=openai/whisper-tiny,STEM=whispertiny,MAX_WORDS=0 \
#     scripts/submit_extract_whisper_lppcn_encoder_wordlocked.sh
#
# lPPCN is Chinese; existing shared features match openai/whisper-tiny (multilingual),
# not whisper-tiny.en. MAX_WORDS=0 means all words.

set -euo pipefail

# Hasson / openai-whisper load_audio uses ffmpeg.
if command -v module >/dev/null 2>&1; then
  module load ffmpeg/5.1.4 >/dev/null 2>&1 || true
fi

# On compute nodes Slurm may run a copy of this script from /var/spool/slurmd/;
# do not derive paths from $0 (that makes mkdir try /var/spool/slurmd/logs).
POOL="${POOL:-/orcd/pool/005/haolun52}"
ROOT_DIR="${ROOT_DIR:-$POOL/dyslexia_natualistics_listing/b2b_decoding/joint_v4}"
SHARED_DIR="${SHARED_DIR:-$POOL/extracted_sections_wordlocked_shared/_shared_wordlocked_features}"
VENV_PY="${VENV_PY:-$POOL/dyslexia_natualistics_listing/b2b_decoding/joint_v4/.venv_gpt2/bin/python}"

SECTION_ID="${SECTION_ID:-1}"
MODEL_NAME="${MODEL_NAME:-openai/whisper-tiny}"
STEM="${STEM:-whispertiny}"
MAX_WORDS="${MAX_WORDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CLIP_END_OFFSET_S="${CLIP_END_OFFSET_S:-0.2325}"
START_WORD_IDX="${START_WORD_IDX:-0}"
NUM_WINDOWS="${NUM_WINDOWS:-10}"
FRAME_MODE="${FRAME_MODE:-hasson}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
SAVE_ALL_ENCODER_LAYERS="${SAVE_ALL_ENCODER_LAYERS:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"

OUTDIR="${OUTDIR:-$POOL/_reextract_whisper_encoder/section_${SECTION_ID}/${STEM}}"

mkdir -p "$OUTDIR"
mkdir -p "$ROOT_DIR/logs"

echo "Submitting extraction with:"
echo "  SECTION_ID=$SECTION_ID"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  STEM=$STEM"
echo "  MAX_WORDS=$MAX_WORDS"
echo "  START_WORD_IDX=$START_WORD_IDX"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  CLIP_END_OFFSET_S=$CLIP_END_OFFSET_S"
echo "  NUM_WINDOWS=$NUM_WINDOWS"
echo "  FRAME_MODE=$FRAME_MODE"
echo "  TORCH_DTYPE=$TORCH_DTYPE"
echo "  SAVE_ALL_ENCODER_LAYERS=$SAVE_ALL_ENCODER_LAYERS"
echo "  OUTDIR=$OUTDIR"

set -x
EXTRA_ARGS=()
if [[ "$SAVE_ALL_ENCODER_LAYERS" == "1" ]]; then
  EXTRA_ARGS+=(--save-all-encoder-layers)
fi
EXTRA_ARGS+=(--frame-mode "$FRAME_MODE")
EXTRA_ARGS+=(--torch-dtype "$TORCH_DTYPE")
"$VENV_PY" "$ROOT_DIR/scripts/extract_whisper_lppcn_encoder_wordlocked.py" \
  --section-id "$SECTION_ID" \
  --shared-dir "$SHARED_DIR" \
  --model-name "$MODEL_NAME" \
  --stem "$STEM" \
  --max-words "$MAX_WORDS" \
  --start-word-idx "$START_WORD_IDX" \
  --batch-size "$BATCH_SIZE" \
  --clip-end-offset-s "$CLIP_END_OFFSET_S" \
  --num-windows "$NUM_WINDOWS" \
  ${HF_CACHE_DIR:+--hf-cache-dir "$HF_CACHE_DIR"} \
  "${EXTRA_ARGS[@]}" \
  --outdir "$OUTDIR"

