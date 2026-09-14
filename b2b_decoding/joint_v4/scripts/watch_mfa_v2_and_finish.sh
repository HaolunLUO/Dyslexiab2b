#!/usr/bin/env bash
# Watch MFA v2 observed array until 63 family_agg files, then submit nulls
# and run group comparison + audit.
#
#   nohup bash scripts/watch_mfa_v2_and_finish.sh > logs/watch_mfa_v2.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

OUT="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_mfa_v2_phone_tone_env_onset_passthrough}"
PYTHON="${B2B_PY:-$(pwd)/.venv_gpt2/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
OBS_JOB="${OBS_JOB:-}"

echo "[$(date -Is)] watching $OUT for 63 family_agg (obs_job=${OBS_JOB:-none})"
stall=0
prev_n=-1
while true; do
  n=$(find "$OUT" -mindepth 2 -maxdepth 2 -name '*_b2b_family_agg.csv' 2>/dev/null | wc -l)
  echo "[$(date -Is)] n_agg=$n"
  if [[ "$n" -ge 63 ]]; then
    break
  fi
  if [[ -n "$OBS_JOB" ]]; then
    left=$(squeue -j "$OBS_JOB" -h 2>/dev/null | wc -l || true)
    echo "  array tasks left≈$left"
  else
    left=0
  fi
  if [[ "$n" -eq "$prev_n" ]]; then
    stall=$((stall + 1))
  else
    stall=0
  fi
  prev_n=$n
  if [[ "$left" -eq 0 && "$stall" -ge 2 ]]; then
    echo "WARN: array idle at $n/63 — looking for missing subjects to resubmit"
    mapfile -t missing < <(OUT="$OUT" "$PYTHON" - <<'PY'
import csv, os
from pathlib import Path
cohort=list(csv.DictReader(open("cohort_groups.csv")))
primary=[r["participant"] for r in cohort if r["include_primary"] in ("1","True","true")]
out=Path(os.environ["OUT"])
have={p.name for p in out.iterdir() if (p/f"{p.name}_b2b_family_agg.csv").is_file()}
for i,p in enumerate(primary,1):
    if p not in have:
        print(i)
PY
)
    if [[ ${#missing[@]} -eq 0 ]]; then
      echo "ERROR: cannot resolve missing subjects" >&2
      exit 1
    fi
    for idx in "${missing[@]}"; do
      echo "Resubmitting array task $idx"
      sbatch --array="$idx" scripts/run_mfa_phone_tone_env_observed_array.sh
    done
    OBS_JOB=""
    stall=0
  fi
  sleep 120
done

echo "[$(date -Is)] observed complete ($n). Submitting null array…"
NULL_JOB=$(sbatch --parsable scripts/run_mfa_phone_tone_env_null_array.sh)
echo "NULL_JOB=$NULL_JOB"

echo "[$(date -Is)] waiting for null aggregation (null_traces.csv × 63)…"
while true; do
  nnull=$(find "$OUT" -mindepth 2 -maxdepth 2 -name '*_null_traces.csv' 2>/dev/null | wc -l)
  echo "[$(date -Is)] n_null_agg=$nnull"
  if [[ "$nnull" -ge 63 ]]; then
    break
  fi
  left=$(squeue -j "$NULL_JOB" -h 2>/dev/null | wc -l || true)
  echo "  null tasks left≈$left"
  if [[ "$left" -eq 0 && "$nnull" -lt 63 ]]; then
    echo "WARNING: null array idle with $nnull/63; attempting aggregate_v4.jl"
    module load julia/1.12.6
    B2B_OUTDIR="$OUT" julia --project=.. aggregate_v4.jl --nulls-only || true
    nnull=$(find "$OUT" -mindepth 2 -maxdepth 2 -name '*_null_traces.csv' 2>/dev/null | wc -l)
    [[ "$nnull" -ge 63 ]] && break
    echo "ERROR: nulls incomplete ($nnull/63)" >&2
    exit 1
  fi
  sleep 180
done

echo "[$(date -Is)] running group comparison (mean_score)…"
bash scripts/run_mfa_phone_tone_env_group_comparison.sh \
  group_comparison_b2b_mfa_v2_phone_tone_env_onset_passthrough

echo "[$(date -Is)] sensitivity mean_trace…"
SCORE_COL=mean_trace bash scripts/run_mfa_phone_tone_env_group_comparison.sh \
  group_comparison_b2b_mfa_v2_mean_trace_sensitivity || true

echo "[$(date -Is)] old-vs-v2 audit…"
"$PYTHON" scripts/audit_mfa_v1_vs_v2.py --out-json "$OUT/old_vs_v2_audit.json"

echo "[$(date -Is)] DONE"
