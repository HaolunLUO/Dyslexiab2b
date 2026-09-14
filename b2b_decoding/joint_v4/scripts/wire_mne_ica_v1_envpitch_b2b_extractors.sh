#!/usr/bin/env bash
# Overlay: MNE-ICA v1 EEG + shared envelope/pitch/onset/frequency/surprisal.
set -euo pipefail

POOL="${POOL:-/orcd/pool/005/haolun52}"
ICA="${ICA:-${POOL}/extracted_sections_wordlocked_mne_ica_v1}"
SHARED="${SHARED:-${POOL}/extracted_sections_wordlocked_shared}"
OVERLAY="${OVERLAY:-${POOL}/extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b}"
COHORT="${COHORT:-$(cd "$(dirname "$0")/.." && pwd)/cohort_groups.csv}"

test -d "$ICA" || { echo "Missing ICA: $ICA"; exit 1; }
test -d "${SHARED}/_shared_wordlocked_features/section_001" || {
  echo "Missing shared features"; exit 1
}
for key in envelope_v2 pitch word_onset offset lexical_frequency gpt2cn_surprisal tone tone_v2 tone_v3; do
  test -f "${SHARED}/_shared_wordlocked_features/section_001/X_word_${key}.npy" || {
    echo "Missing X_word_${key}.npy — run extract_pitch_onset_wordlocked.py / extract_tone_phoneme_wordlocked.py / extract_tone_v2_wordlocked.py / extract_tone_v3_wordlocked.py / prepare_lexical_features.jl"
    exit 1
  }
done

mapfile -t SUBJECTS < <(python3 - <<PY
import csv
rows = [r for r in csv.DictReader(open("$COHORT"))
        if r["include_primary"] in ("1", "True", "true")]
print("\n".join(r["participant"] for r in rows))
PY
)

echo "Wiring ICA v1 env/pitch/onset overlay → $OVERLAY  n=${#SUBJECTS[@]}"
mkdir -p "$OVERLAY"
ln -sfn "${SHARED}/_shared_wordlocked_features" "${OVERLAY}/_shared_wordlocked_features"
n_ok=0
for subj in "${SUBJECTS[@]}"; do
  test -d "${ICA}/${subj}" || { echo "Missing ${ICA}/${subj}"; exit 1; }
  ln -sfn "${ICA}/${subj}" "${OVERLAY}/${subj}"
  n_ok=$((n_ok + 1))
done
echo "Done. wired ${n_ok}. export B2B_EXTRACTOR_DIR=$OVERLAY"
