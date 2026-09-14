#!/usr/bin/env bash
# Overlay: existing ICA EEG + NEW nucleus-locked shared features.
# Does not retarget extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b.
set -euo pipefail

POOL="${POOL:-/orcd/pool/005/haolun52}"
ICA="${ICA:-${POOL}/extracted_sections_wordlocked_mne_ica_v1}"
SHARED="${SHARED:-${POOL}/extracted_sections_nucleuslocked_shared}"
OVERLAY="${OVERLAY:-${POOL}/extracted_sections_nucleuslocked_mne_ica_v1_b2b}"
COHORT="${COHORT:-$(cd "$(dirname "$0")/.." && pwd)/cohort_groups.csv}"

test -d "$ICA" || { echo "Missing ICA: $ICA"; exit 1; }
test -d "${SHARED}/_shared_wordlocked_features/section_001" || {
  echo "Missing nucleus-locked shared features"; exit 1
}
for key in ctx_controls ctx_pitch ctx_evidence ctx_resid; do
  test -f "${SHARED}/_shared_wordlocked_features/section_001/X_word_${key}.npy" || {
    echo "Missing X_word_${key}.npy — run extract_tone_ctx_v1_nucleuslocked.py"
    exit 1
  }
done
# Refuse to clobber the frozen word-locked overlay.
FROZEN="${POOL}/extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b"
if [[ "$(realpath -m "$OVERLAY")" == "$(realpath -m "$FROZEN")" ]]; then
  echo "Refusing to wire into the frozen word-locked overlay"; exit 1
fi

mapfile -t SUBJECTS < <(python3 - <<PY
import csv
rows = [r for r in csv.DictReader(open("$COHORT"))
        if r["include_primary"] in ("1", "True", "true")]
print("\n".join(r["participant"] for r in rows))
PY
)

echo "Wiring ICA EEG + nucleus-locked features → $OVERLAY  n=${#SUBJECTS[@]}"
mkdir -p "$OVERLAY"
ln -sfn "${SHARED}/_shared_wordlocked_features" "${OVERLAY}/_shared_wordlocked_features"
n_ok=0
for subj in "${SUBJECTS[@]}"; do
  test -d "${ICA}/${subj}" || { echo "Missing ${ICA}/${subj}"; exit 1; }
  # Do not symlink the whole subject dir — that would expose word_events.csv
  # (1753 word-locked rows) and override nucleus timing.
  for sid in 001 002; do
    src="${ICA}/${subj}/section_${sid}"
    dst="${OVERLAY}/${subj}/section_${sid}"
    mkdir -p "$dst"
    for f in eeg_data.npy metadata.json train_keep_mask.npy; do
      if [[ -e "${src}/${f}" ]]; then
        ln -sfn "${src}/${f}" "${dst}/${f}"
      fi
    done
  done
  n_ok=$((n_ok + 1))
done
echo "Done. wired ${n_ok}. export B2B_EXTRACTOR_DIR=$OVERLAY"
