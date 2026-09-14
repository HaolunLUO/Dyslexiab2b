#!/usr/bin/env bash
# Orchestrate MFA v2 full cohort after a successful pilot.
#
#   bash scripts/submit_mfa_v2_full_cohort.sh
#
# Submits: observed array (63) → (operator re-runs nulls + group comparison
# after observed completes). Prints the follow-up commands.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export B2B_OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_mfa_v2_phone_tone_env_onset_passthrough}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_FEATURE_SETS="${B2B_FEATURE_SETS:-envelope_v2,mfa_phones_v2,mfa_tone_v2}"
export B2B_FAMILY_NAMES="${B2B_FAMILY_NAMES:-envelope,phones,tone}"

SHARED="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared}/_shared_wordlocked_features"
PYTHON="${B2B_PY:-$(pwd)/.venv_gpt2/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3

echo "Preflight…"
"$PYTHON" scripts/preflight_mfa_v2_features.py --shared-dir "$SHARED" --keys "$B2B_FEATURE_SETS"

if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "ERROR: missing basis at $B2B_BASIS_DIR — run pilot first to prepare basis." >&2
  exit 1
fi

OBS_JOB=$(sbatch --parsable scripts/run_mfa_phone_tone_env_observed_array.sh)
echo "Submitted observed array: $OBS_JOB"
echo
echo "After observed finishes (63 family_agg files), run:"
echo "  sbatch scripts/run_mfa_phone_tone_env_null_array.sh"
echo "  bash scripts/run_mfa_phone_tone_env_group_comparison.sh"
echo "  $PYTHON scripts/audit_mfa_v1_vs_v2.py --out-json \$B2B_OUTDIR/old_vs_v2_audit.json"
echo
echo "Sensitivity (mean_trace):"
echo "  SCORE_COL=mean_trace OUT=group_comparison_b2b_mfa_v2_mean_trace_sensitivity \\"
echo "    bash -c 'SCORE_COL=mean_trace bash scripts/run_mfa_phone_tone_env_group_comparison.sh group_comparison_b2b_mfa_v2_mean_trace_sensitivity'"
