#!/usr/bin/env bash
# Overlay extractor for Hasson-aligned Whisper large-v3 B2B runs.
#
# Keeps cohort EEG / word_events from the shared extractor; replaces only the
# four whisperlargev3 word-locked matrices (+ sidecars) per section.
#
# Usage: bash scripts/wire_whisperlargev3_b2b_extractors.sh
set -euo pipefail

POOL="${POOL:-/orcd/pool/005/haolun52}"
SHARED="${SHARED:-${POOL}/extracted_sections_wordlocked_shared}"
OVERLAY="${OVERLAY:-${POOL}/extracted_sections_wordlocked_largev3_b2b}"
ENC_ROOT="${ENC_ROOT:-${POOL}/_reextract_whisper_encoder}"
LANG_ROOT="${LANG_ROOT:-${POOL}/_reextract_whisper_language}"
COHORT="${COHORT:-$(cd "$(dirname "$0")/.." && pwd)/cohort_groups.csv}"

test -d "$SHARED" || { echo "Missing shared extractor: $SHARED"; exit 1; }
test -f "$COHORT" || { echo "Missing cohort: $COHORT"; exit 1; }

mapfile -t SUBJECTS < <(python3 - <<PY
import csv
rows = [r for r in csv.DictReader(open("$COHORT"))
        if r["include_primary"] in ("1", "True", "true")]
print("\n".join(r["participant"] for r in rows))
PY
)

echo "Wiring large-v3 B2B overlay → $OVERLAY"
echo "  shared  : $SHARED"
echo "  encoder : $ENC_ROOT"
echo "  language: $LANG_ROOT"
echo "  subjects: ${#SUBJECTS[@]}"

mkdir -p "$OVERLAY"

for subj in "${SUBJECTS[@]}"; do
  test -d "${SHARED}/${subj}" || { echo "Missing participant dir: ${SHARED}/${subj}"; exit 1; }
  ln -sfn "${SHARED}/${subj}" "${OVERLAY}/${subj}"
done

link_variant() {
  local dst_dir="$1"
  local src_dir="$2"
  local variant="$3"
  for name in \
    "X_word_whisperlargev3_${variant}.npy" \
    "X_word_whisperlargev3_${variant}_feature_names.txt" \
    "X_word_whisperlargev3_${variant}_meta.json"
  do
    local src="${src_dir}/${name}"
    test -f "$src" || { echo "Missing $src"; exit 1; }
    ln -sfn "$src" "${dst_dir}/${name}"
  done
}

wire_section() {
  local sec_tag="$1"
  local sec_num="$2"
  local dst="${OVERLAY}/_shared_wordlocked_features/${sec_tag}"
  local src_shared="${SHARED}/_shared_wordlocked_features/${sec_tag}"
  local src_enc="${ENC_ROOT}/section_${sec_num}/whisperlargev3_hasson_v1"
  local src_lang="${LANG_ROOT}/section_${sec_num}/whisperlargev3_hasson_v1"

  test -d "$src_shared" || { echo "Missing $src_shared"; exit 1; }
  test -d "$src_enc" || { echo "Missing $src_enc"; exit 1; }
  test -d "$src_lang" || { echo "Missing $src_lang"; exit 1; }

  mkdir -p "$dst"
  for f in "$src_shared"/*; do
    ln -sfn "$f" "${dst}/$(basename "$f")"
  done

  link_variant "$dst" "$src_enc" acoustic
  link_variant "$dst" "$src_enc" speech
  link_variant "$dst" "$src_lang" language_audio_fused
  link_variant "$dst" "$src_lang" language_text_only

  echo "  wired ${sec_tag}"
}

wire_section section_001 1
wire_section section_002 2

echo "Done. export B2B_EXTRACTOR_DIR=$OVERLAY"
