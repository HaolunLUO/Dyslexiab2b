#!/usr/bin/env bash
# Shared env for tone_ctx_v1 nested B2B. Source from submit scripts.
# Usage: MODEL=M0|M1|M2|M3 source scripts/tone_ctx_v1_env.sh
set -euo pipefail
POOL="${POOL:-/orcd/pool/005/haolun52}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:?MODEL=M0|M1|M2|M3}"
LENGTHS="${ROOT}/tone_ctx_v1/sec_lengths.json"
test -f "$LENGTHS" || { echo "Missing $LENGTHS — run extract_tone_ctx_v1_nucleuslocked.py"; exit 1; }
# Colon, not comma: sbatch --export splits on commas, so "2372,2590" became just 2372.
eval "$(python3 - <<PY
import json
d = json.load(open("$LENGTHS"))
n1, n2 = int(d["1"]), int(d["2"])
print(f"export B2B_EXPECTED_SEC_LENGTHS={n1}:{n2}")
print(f"export B2B_EXPECTED_N_WORDS={n1+n2}")
PY
)"
export B2B_EXTRACTOR_DIR="${POOL}/extracted_sections_nucleuslocked_mne_ica_v1_b2b"
export B2B_EPOCH_ANCHOR=offset
export B2B_TMIN_S=-0.2
export B2B_TMAX_S=0.8
export B2B_BASIS_MODE=independent
export B2B_PASSTHROUGH=1
export B2B_H_RIDGE_KAPPA=0
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_STRICT_FEAT_DIMS=0
export B2B_NO_PLOT=1
case "$MODEL" in
  M0)
    export B2B_FEATURE_SETS=ctx_controls
    export B2B_FAMILY_NAMES=controls
    ;;
  M1)
    export B2B_FEATURE_SETS=ctx_controls:ctx_pitch
    export B2B_FAMILY_NAMES=controls:pitch
    ;;
  M2)
    export B2B_FEATURE_SETS=ctx_controls:ctx_pitch:ctx_evidence
    export B2B_FAMILY_NAMES=controls:pitch:evidence
    ;;
  M3)
    export B2B_FEATURE_SETS=ctx_controls:ctx_pitch:ctx_evidence:ctx_resid
    export B2B_FAMILY_NAMES=controls:pitch:evidence:resid
    ;;
  *) echo "MODEL must be M0 M1 M2 M3"; exit 1 ;;
esac
export B2B_OUTDIR="${POOL}/encoding_results_b2b_tone_ctx_v1_${MODEL}_nucleus_offset"
export B2B_BASIS_DIR="${B2B_OUTDIR}/_basis"
