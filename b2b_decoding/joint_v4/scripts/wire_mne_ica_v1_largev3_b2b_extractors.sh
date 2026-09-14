#!/usr/bin/env bash
# Overlay extractor for MNE-ICA v1 EEG + Hasson-aligned Whisper large-v3 features.
#
# Per-subject EEG / word_events / masks come from extracted_sections_wordlocked_mne_ica_v1.
# Shared word-locked Whisper large-v3 matrices come from the existing large-v3 overlay.
# Does not modify the uploaded ICA tree.
#
# Usage: bash scripts/wire_mne_ica_v1_largev3_b2b_extractors.sh
set -euo pipefail

POOL="${POOL:-/orcd/pool/005/haolun52}"
ICA="${ICA:-${POOL}/extracted_sections_wordlocked_mne_ica_v1}"
FEAT_OVERLAY="${FEAT_OVERLAY:-${POOL}/extracted_sections_wordlocked_largev3_b2b}"
OVERLAY="${OVERLAY:-${POOL}/extracted_sections_wordlocked_mne_ica_v1_largev3_b2b}"
COHORT="${COHORT:-$(cd "$(dirname "$0")/.." && pwd)/cohort_groups.csv}"

test -d "$ICA" || { echo "Missing ICA extractor: $ICA"; exit 1; }
test -d "${FEAT_OVERLAY}/_shared_wordlocked_features/section_001" || {
  echo "Missing large-v3 shared features under $FEAT_OVERLAY"
  echo "Run: bash scripts/wire_whisperlargev3_b2b_extractors.sh"
  exit 1
}
test -f "$COHORT" || { echo "Missing cohort: $COHORT"; exit 1; }

mapfile -t SUBJECTS < <(python3 - <<PY
import csv
rows = [r for r in csv.DictReader(open("$COHORT"))
        if r["include_primary"] in ("1", "True", "true")]
print("\n".join(r["participant"] for r in rows))
PY
)

echo "Wiring MNE-ICA v1 + large-v3 B2B overlay → $OVERLAY"
echo "  ica     : $ICA"
echo "  feats   : $FEAT_OVERLAY/_shared_wordlocked_features"
echo "  subjects: ${#SUBJECTS[@]}"

mkdir -p "$OVERLAY"
ln -sfn "${FEAT_OVERLAY}/_shared_wordlocked_features" \
  "${OVERLAY}/_shared_wordlocked_features"

n_ok=0
for subj in "${SUBJECTS[@]}"; do
  src="${ICA}/${subj}"
  test -d "$src" || { echo "Missing participant dir: $src"; exit 1; }
  for sec in section_001 section_002; do
    test -f "${src}/${sec}/eeg_data.npy" || {
      echo "Missing EEG: ${src}/${sec}/eeg_data.npy"; exit 1;
    }
    test -f "${src}/${sec}/word_events.csv" || {
      echo "Missing word_events: ${src}/${sec}/word_events.csv"; exit 1;
    }
  done
  ln -sfn "$src" "${OVERLAY}/${subj}"
  n_ok=$((n_ok + 1))
done

echo "Done. wired ${n_ok} subjects. export B2B_EXTRACTOR_DIR=$OVERLAY"
