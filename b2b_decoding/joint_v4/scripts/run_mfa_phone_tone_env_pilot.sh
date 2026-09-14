#!/bin/bash
# Onset-locked B2B pilot: envelope_v2 + MFA Mandarin phones_v2 + tone_v2.
# Same word onsets as Whisper / lexical arms. OLS H (no ridge-H fallback).
#
# Usage:
#   bash scripts/run_mfa_phone_tone_env_pilot.sh           # RN109 / D007d / D011d
#   bash scripts/run_mfa_phone_tone_env_pilot.sh RN109
#
# Requires v2 features from extract_mfa_phone_tone_envelope_wordlocked.py

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}"

export B2B_MODE=observed
export B2B_EPOCH_ANCHOR=onset
export B2B_BASIS_MODE=independent
# Interpretable MFA features: keep columns; require identifiable OLS H.
export B2B_PASSTHROUGH=1
export B2B_H_RIDGE_KAPPA=0
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_NO_PLOT=1
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-4}"
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_STRICT_FEAT_DIMS=0

export B2B_FEATURE_SETS="${B2B_FEATURE_SETS:-envelope_v2,mfa_phones_v2,mfa_tone_v2}"
export B2B_FAMILY_NAMES="${B2B_FAMILY_NAMES:-envelope,phones,tone}"
export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_mfa_v2_phone_tone_env_onset_passthrough}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"

SHARED="$B2B_EXTRACTOR_DIR/_shared_wordlocked_features"
PYTHON="${B2B_PY:-${PWD}/.venv_gpt2/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3

IFS=',;:' read -r -a FEATS <<< "$B2B_FEATURE_SETS"
for f in "${FEATS[@]}"; do
  f="${f// /}"
  [[ -z "$f" ]] && continue
  if [[ ! -f "$SHARED/section_001/X_word_${f}.npy" ]]; then
    echo "ERROR: missing $SHARED/section_001/X_word_${f}.npy"
    echo "Submit MFA v2 extraction first:"
    echo "  mkdir -p logs && sbatch scripts/submit_extract_mfa_phone_tone_envelope.sh"
    exit 1
  fi
done

echo "=== MFA v2 preflight QC ==="
"$PYTHON" scripts/preflight_mfa_v2_features.py \
  --shared-dir "$SHARED" \
  --keys "$B2B_FEATURE_SETS" \
  --out-json "$SHARED/mfa_v2_preflight_qc.json"

module load julia/1.12.6

echo "=== MFA phone / tone / envelope B2B v2 pilot ==="
echo "  outdir   : $B2B_OUTDIR"
echo "  features : $B2B_FEATURE_SETS"
echo "  families : $B2B_FAMILY_NAMES"
echo "  passthrough=$B2B_PASSTHROUGH  H_ridge_kappa=$B2B_H_RIDGE_KAPPA  anchor=$B2B_EPOCH_ANCHOR"

if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" || "${B2B_FORCE_BASIS:-0}" == "1" ]]; then
  echo "Preparing feature basis…"
  mkdir -p "$B2B_BASIS_DIR"
  julia --project=.. prepare_feature_basis.jl
fi

SUBJECTS=("$@")
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  SUBJECTS=(RN109 D007d D011d)
fi

for p in "${SUBJECTS[@]}"; do
  echo "======== MFA v2 PHONE/TONE/ENV $p ========"
  julia --project=.. b2b_joint_v4_pipeline.jl "$p" 2>&1 | tee "logs/mfa_v2_phone_tone_env_${p}.out"
  # Pilot gates: 20/20 valid partitions, finite scores, OLS H
  "$PYTHON" - "$B2B_OUTDIR" "$p" <<'PY'
import csv, math, sys
from pathlib import Path
outdir, p = Path(sys.argv[1]), sys.argv[2]
qc = outdir / p / f"{p}_qc_summary.csv"
diag = outdir / p / f"{p}_diagnostics.csv"
agg = outdir / p / f"{p}_b2b_family_agg.csv"
assert qc.is_file() and diag.is_file() and agg.is_file(), f"missing outputs for {p}"
with qc.open() as f:
    row = next(csv.DictReader(f))
n_valid = int(float(row.get("n_valid_partitions", -1)))
if n_valid < 20:
    raise SystemExit(f"PILOT FAIL {p}: n_valid_partitions={n_valid} < 20")
bad = 0
with diag.open() as f:
    for r in csv.DictReader(f):
        if str(r.get("direction", "0")) not in ("0", "0.0"):
            continue
        if str(r.get("invalid", "0")) in ("1", "True", "true"):
            bad += 1
        try:
            k = float(r.get("max_kappa", "nan"))
        except ValueError:
            k = float("nan")
        if not math.isfinite(k):
            bad += 1
if bad:
    raise SystemExit(f"PILOT FAIL {p}: {bad} invalid/nonfinite diagnostic rows")
with agg.open() as f:
    for r in csv.DictReader(f):
        for col in ("mean_score", "mean_trace"):
            if col in r and r[col] not in ("", "NA"):
                v = float(r[col])
                if not math.isfinite(v):
                    raise SystemExit(f"PILOT FAIL {p}: nonfinite {col}")
# Require OLS H in run manifest
import json
man = json.loads((outdir / p / f"{p}_run_manifest.json").read_text())
hk = float(man.get("h_ridge_kappa", -1))
if hk != 0.0:
    raise SystemExit(f"PILOT FAIL {p}: h_ridge_kappa={hk} (expected 0 OLS)")
print(f"PILOT OK {p}: n_valid={n_valid} max_kappa_ok OLS_H")
PY
done

# Optional small stimulus-shift null smoke (1 subject, 5 reps) before scaling.
# Uses a nested outdir because mode=null changes the analysis-manifest fingerprint.
if [[ "${B2B_PILOT_NULL_SMOKE:-1}" == "1" ]]; then
  echo "=== Null smoke (5 stimulus-shift reps on ${SUBJECTS[0]}) ==="
  MAIN_OUT="$B2B_OUTDIR"
  MAIN_BASIS="${B2B_BASIS_DIR:-${MAIN_OUT}/_basis}"
  SMOKE_OUT="${MAIN_OUT}/_null_smoke"
  mkdir -p "${SMOKE_OUT}/${SUBJECTS[0]}"
  # Reuse observed artifacts so null mode does not re-fit G/H.
  if [[ -d "${MAIN_OUT}/${SUBJECTS[0]}" ]]; then
    cp -a "${MAIN_OUT}/${SUBJECTS[0]}/." "${SMOKE_OUT}/${SUBJECTS[0]}/"
  fi
  (
    export B2B_MODE=null
    export B2B_N_NULL=5
    export B2B_NULL_START=1
    export B2B_NULL_COUNT=5
    export B2B_OUTDIR="$SMOKE_OUT"
    export B2B_BASIS_DIR="$MAIN_BASIS"
    julia --project=.. b2b_joint_v4_pipeline.jl "${SUBJECTS[0]}"
  ) 2>&1 | tee "logs/mfa_v2_null_smoke_${SUBJECTS[0]}.out"
  smoke_csv="${SMOKE_OUT}/${SUBJECTS[0]}/${SUBJECTS[0]}_null_0001_0005_traces.csv"
  [[ -f "$smoke_csv" ]] || { echo "ERROR: null smoke missing $smoke_csv" >&2; exit 1; }
  echo "Null smoke OK → $smoke_csv"
fi

echo "MFA v2 phone/tone/envelope pilot complete → $B2B_OUTDIR"
