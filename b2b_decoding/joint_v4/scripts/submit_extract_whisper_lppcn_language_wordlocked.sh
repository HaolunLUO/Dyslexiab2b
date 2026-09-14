#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=logs/extract_whisper_language_%j.out
#SBATCH --error=logs/extract_whisper_language_%j.err
#
# Submit a GPU job to re-extract Whisper decoder language embeddings
# (language_audio_fused + language_text_only) from lPPCN section WAVs.
#
# Example smoke:
#   sbatch --export=ALL,SECTION_ID=1,MAX_WORDS=20,START_WORD_IDX=101,PROBE_ALL_DECODER_LAYERS=1 \
#     scripts/submit_extract_whisper_lppcn_language_wordlocked.sh

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load ffmpeg/5.1.4 >/dev/null 2>&1 || true
fi

POOL="${POOL:-/orcd/pool/005/haolun52}"
ROOT_DIR="${ROOT_DIR:-$POOL/dyslexia_natualistics_listing/b2b_decoding/joint_v4}"
SHARED_DIR="${SHARED_DIR:-$POOL/extracted_sections_wordlocked_shared/_shared_wordlocked_features}"
VENV_PY="${VENV_PY:-$POOL/dyslexia_natualistics_listing/b2b_decoding/joint_v4/.venv_gpt2/bin/python}"

SECTION_ID="${SECTION_ID:-1}"
MODEL_NAME="${MODEL_NAME:-openai/whisper-tiny}"
STEM="${STEM:-whispertiny}"
MAX_WORDS="${MAX_WORDS:-0}"
START_WORD_IDX="${START_WORD_IDX:-0}"
CLIP_END_OFFSET_S="${CLIP_END_OFFSET_S:-0.2325}"
WINDOW_MODE="${WINDOW_MODE:-full-onset}"
DECODER_LAYER="${DECODER_LAYER:-3}"
# Comma-separated lists break sbatch --export; use colons, e.g. 0:3:8:16:24:32
DECODER_LAYERS="${DECODER_LAYERS:-}"
PROBE_ALL_DECODER_LAYERS="${PROBE_ALL_DECODER_LAYERS:-0}"
TOKEN_POOL="${TOKEN_POOL:-last}"
PREFIX_STYLE="${PREFIX_STYLE:-zh-transcribe}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"

OUTDIR="${OUTDIR:-$POOL/_reextract_whisper_language/section_${SECTION_ID}/${STEM}}"

mkdir -p "$OUTDIR"
mkdir -p "$ROOT_DIR/logs"

echo "Submitting language extraction with:"
echo "  SECTION_ID=$SECTION_ID"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  STEM=$STEM"
echo "  MAX_WORDS=$MAX_WORDS"
echo "  START_WORD_IDX=$START_WORD_IDX"
echo "  WINDOW_MODE=$WINDOW_MODE"
echo "  CLIP_END_OFFSET_S=$CLIP_END_OFFSET_S"
echo "  DECODER_LAYER=$DECODER_LAYER"
echo "  DECODER_LAYERS=${DECODER_LAYERS:-}"
echo "  PROBE_ALL_DECODER_LAYERS=$PROBE_ALL_DECODER_LAYERS"
echo "  TOKEN_POOL=$TOKEN_POOL"
echo "  PREFIX_STYLE=$PREFIX_STYLE"
echo "  OUTDIR=$OUTDIR"

EXTRA_ARGS=()
if [[ -n "$DECODER_LAYER" ]]; then
  EXTRA_ARGS+=(--decoder-layer "$DECODER_LAYER")
fi
if [[ -n "$DECODER_LAYERS" ]]; then
  EXTRA_ARGS+=(--decoder-layers "$DECODER_LAYERS")
fi
if [[ "$PROBE_ALL_DECODER_LAYERS" == "1" ]]; then
  EXTRA_ARGS+=(--probe-all-decoder-layers)
fi
EXTRA_ARGS+=(--token-pool "$TOKEN_POOL")
EXTRA_ARGS+=(--prefix-style "$PREFIX_STYLE")

set -x
"$VENV_PY" "$ROOT_DIR/scripts/extract_whisper_lppcn_language_wordlocked.py" \
  --section-id "$SECTION_ID" \
  --shared-dir "$SHARED_DIR" \
  --model-name "$MODEL_NAME" \
  --stem "$STEM" \
  --max-words "$MAX_WORDS" \
  --start-word-idx "$START_WORD_IDX" \
  --clip-end-offset-s "$CLIP_END_OFFSET_S" \
  --window-mode "$WINDOW_MODE" \
  ${HF_CACHE_DIR:+--hf-cache-dir "$HF_CACHE_DIR"} \
  "${EXTRA_ARGS[@]}" \
  --outdir "$OUTDIR"
