#!/usr/bin/env bash
# Build B2B-ready extractor overlays for banded EEG.
#
# Banded trees only ship eeg_data.npy + metadata.json. B2B also needs
# word_events.csv, train_keep_mask.npy, and _shared_wordlocked_features.
# This script creates a parallel overlay tree of symlinks — it does not
# modify the banded source.
#
# Usage: bash scripts/wire_banded_b2b_extractors.sh
set -euo pipefail

POOL="${POOL:-/orcd/pool/005/haolun52}"
BANDED="${BANDED:-${POOL}/extracted_sections_wordlocked_shared_banded}"
SHARED="${SHARED:-${POOL}/extracted_sections_wordlocked_shared}"
OVERLAY_ROOT="${OVERLAY_ROOT:-${POOL}/extracted_sections_wordlocked_banded_b2b}"
COHORT="${COHORT:-$(cd "$(dirname "$0")/.." && pwd)/cohort_groups.csv}"
BANDS="${BANDS:-delta theta alpha beta broadband}"

test -d "$BANDED" || { echo "Missing banded root: $BANDED"; exit 1; }
test -d "$SHARED/_shared_wordlocked_features" || {
  echo "Missing shared features: $SHARED/_shared_wordlocked_features"; exit 1;
}
test -f "$COHORT" || { echo "Missing cohort: $COHORT"; exit 1; }

mapfile -t SUBJECTS < <(python3 - <<PY
import csv
rows = [r for r in csv.DictReader(open("$COHORT"))
        if r["include_primary"] in ("1", "True", "true")]
print("\n".join(r["participant"] for r in rows))
PY
)

echo "Wiring ${#SUBJECTS[@]} cohort subjects × bands: $BANDS"
echo "  banded  : $BANDED"
echo "  shared  : $SHARED"
echo "  overlay : $OVERLAY_ROOT"

for band in $BANDS; do
  src="${BANDED}/${band}"
  dst="${OVERLAY_ROOT}/${band}"
  test -d "$src" || { echo "Missing band dir: $src"; exit 1; }
  mkdir -p "$dst"
  ln -sfn "$SHARED/_shared_wordlocked_features" "$dst/_shared_wordlocked_features"

  n_ok=0
  for subj in "${SUBJECTS[@]}"; do
    for sec in section_001 section_002; do
      sdir="${src}/${subj}/${sec}"
      ddir="${dst}/${subj}/${sec}"
      test -f "${sdir}/eeg_data.npy" || {
        echo "Missing EEG: ${sdir}/eeg_data.npy"; exit 1;
      }
      test -f "${SHARED}/${subj}/${sec}/word_events.csv" || {
        echo "Missing word_events: ${SHARED}/${subj}/${sec}/word_events.csv"; exit 1;
      }
      test -f "${SHARED}/${subj}/${sec}/train_keep_mask.npy" || {
        echo "Missing mask: ${SHARED}/${subj}/${sec}/train_keep_mask.npy"; exit 1;
      }
      mkdir -p "$ddir"
      ln -sfn "${sdir}/eeg_data.npy" "${ddir}/eeg_data.npy"
      ln -sfn "${sdir}/metadata.json" "${ddir}/metadata.json"
      ln -sfn "${SHARED}/${subj}/${sec}/word_events.csv" "${ddir}/word_events.csv"
      ln -sfn "${SHARED}/${subj}/${sec}/train_keep_mask.npy" "${ddir}/train_keep_mask.npy"
    done
    n_ok=$((n_ok + 1))
  done
  echo "  [${band}] wired ${n_ok} subjects → ${dst}"
done

echo "Done."
