#!/usr/bin/env bash
# Provenance-aware skip for MFA v2 observed jobs.
#
# Skip only when:
#   1. family_agg CSV exists
#   2. run_manifest.json exists
#   3. feature_basis_hash matches current B2B_BASIS_DIR/feature_basis_meta.json
#   4. every feature checksum in the manifest matches the current X_word_*.npy files
#
# Otherwise: exit 1 (stale) so the caller recomputes, unless B2B_FORCE=1.
#
# Usage:
#   source scripts/mfa_v2_provenance.sh
#   mfa_v2_should_skip_subject "$participant" && exit 0

mfa_v2_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

mfa_v2_should_skip_subject() {
  local participant="$1"
  local outdir="${B2B_OUTDIR:?}"
  local basis_dir="${B2B_BASIS_DIR:-${outdir}/_basis}"
  local extractor="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"
  local shared="${extractor}/_shared_wordlocked_features"
  local feats="${B2B_FEATURE_SETS:-envelope_v2,mfa_phones_v2,mfa_tone_v2}"
  local agg="${outdir}/${participant}/${participant}_b2b_family_agg.csv"
  local man="${outdir}/${participant}/${participant}_run_manifest.json"

  if [[ "${B2B_FORCE:-0}" == "1" ]]; then
    return 1
  fi
  if [[ ! -f "$agg" || ! -f "$man" ]]; then
    return 1
  fi

  local py="${B2B_PY:-python3}"
  "$py" - "$man" "$basis_dir" "$shared" "$feats" <<'PY'
import hashlib, json, sys
from pathlib import Path

man_path, basis_dir, shared, feats = sys.argv[1:5]
man = json.loads(Path(man_path).read_text(encoding="utf-8"))
meta_path = Path(basis_dir) / "feature_basis_meta.json"
if not meta_path.is_file():
    print(f"STALE: missing basis meta {meta_path}")
    sys.exit(2)
basis_hash = hashlib.sha256(meta_path.read_bytes()).hexdigest()
stored_basis = str(man.get("feature_basis_hash", ""))
if stored_basis != basis_hash:
    print(f"STALE: basis hash mismatch stored={stored_basis[:12]}… current={basis_hash[:12]}…")
    sys.exit(2)

stored = man.get("feature_checksums") or {}
keys = [k.strip() for k in feats.replace(";", ",").replace(":", ",").split(",") if k.strip()]
for sid in (1, 2):
    sec = f"section_{sid:03d}"
    for feat in keys:
        path = Path(shared) / sec / f"X_word_{feat}.npy"
        if not path.is_file():
            print(f"STALE: missing feature {path}")
            sys.exit(2)
        cur = hashlib.sha256(path.read_bytes()).hexdigest()
        key = f"{sec}/{feat}"
        # Julia manifest may nest as section_001/feat
        got = stored.get(key) or stored.get(feat)
        if got is None:
            print(f"STALE: manifest missing checksum for {key}")
            sys.exit(2)
        if str(got) != cur:
            print(f"STALE: checksum mismatch for {key}")
            sys.exit(2)
print("PROVENANCE_OK")
sys.exit(0)
PY
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[skip] $participant (agg + matching checksums/basis)"
    return 0
  fi
  if [[ $rc -eq 2 ]]; then
    echo "[recompute] $participant (stale or mismatched provenance)"
    return 1
  fi
  echo "ERROR: provenance check failed for $participant (rc=$rc)" >&2
  return 1
}
