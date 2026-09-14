#!/usr/bin/env bash
# Watch MNE-ICA v1 large-v3 observed array, then submit nulls and group-compare.
#
#   OBS_JOB=<id> nohup bash scripts/watch_mne_ica_v1_largev3_and_finish.sh > logs/watch_mne_ica_v1_onset.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
if [[ "$ANCHOR" == "offset" ]]; then
  OUT="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_offset_joint_k8_mne_ica_v1}"
else
  OUT="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_largev3_onset_joint_k8_mne_ica_v1}"
fi
export B2B_OUTDIR="$OUT"
export B2B_EPOCH_ANCHOR="$ANCHOR"
PYTHON="${B2B_PY:-$(pwd)/.venv_gpt2/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
OBS_JOB="${OBS_JOB:-}"

echo "[$(date -Is)] watching $OUT for 63 family_agg (obs_job=${OBS_JOB:-none} anchor=$ANCHOR)"
stall=0
prev_n=-1
while true; do
  n=$(find "$OUT" -mindepth 2 -maxdepth 2 -name '*_b2b_family_agg.csv' 2>/dev/null | wc -l)
  echo "[$(date -Is)] n_agg=$n"
  if [[ "$n" -ge 63 ]]; then
    break
  fi
  left=0
  if [[ -n "$OBS_JOB" ]]; then
    left=$(squeue -j "$OBS_JOB" -h 2>/dev/null | wc -l || true)
    echo "  array tasks left≈$left"
  fi
  if [[ "$n" -eq "$prev_n" ]]; then
    stall=$((stall + 1))
  else
    stall=0
  fi
  prev_n=$n
  if [[ "$left" -eq 0 && "$stall" -ge 3 && "$n" -gt 0 ]]; then
    echo "WARN: array idle at $n/63 — resubmitting missing observed tasks"
    mapfile -t missing < <(OUT="$OUT" "$PYTHON" - <<'PY'
import csv, os
from pathlib import Path
cohort=list(csv.DictReader(open("cohort_groups.csv")))
primary=[r["participant"] for r in cohort if r["include_primary"] in ("1","True","true")]
out=Path(os.environ["OUT"])
have={p.name for p in out.iterdir() if p.is_dir() and (p/f"{p.name}_b2b_family_agg.csv").is_file()}
for i,p in enumerate(primary,1):
    if p not in have:
        print(i)
PY
)
    if [[ ${#missing[@]} -eq 0 ]]; then
      echo "ERROR: cannot resolve missing subjects" >&2
      exit 1
    fi
    spec=$(IFS=,; echo "${missing[*]}")
    echo "Resubmitting array tasks $spec"
    EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-${POOL}/.julia}"
    EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_mne_ica_v1_largev3_b2b}"
    EXPORT+=",B2B_OUTDIR=${OUT}"
    EXPORT+=",B2B_BASIS_DIR=${OUT}/_basis"
    EXPORT+=",B2B_MODE=observed"
    EXPORT+=",B2B_PCA_K=8"
    EXPORT+=",B2B_BASIS_MODE=independent"
    EXPORT+=",B2B_EPOCH_ANCHOR=${ANCHOR}"
    EXPORT+=",B2B_FEAT_ACOUSTIC=whisperlargev3_acoustic"
    EXPORT+=",B2B_FEAT_SPEECH=whisperlargev3_speech"
    EXPORT+=",B2B_FEAT_LANGUAGE=whisperlargev3_language_audio_fused"
    EXPORT+=",B2B_NO_PLOT=1"
    OBS_JOB=$(sbatch --parsable --array="$spec" --export="$EXPORT" scripts/run_observed_array.sh)
    stall=0
  fi
  sleep 180
done

echo "[$(date -Is)] observed complete ($n). Submitting null array…"
NULL_JOB=$(B2B_EPOCH_ANCHOR="$ANCHOR" B2B_OUTDIR="$OUT" bash scripts/submit_mne_ica_v1_largev3_null_array.sh | tail -n 1)
echo "NULL_JOB=$NULL_JOB"

echo "[$(date -Is)] waiting for null aggregation…"
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
    echo "after aggregate n_null_agg=$nnull"
    if [[ "$nnull" -lt 63 ]]; then
      echo "ERROR: nulls incomplete ($nnull/63)" >&2
      exit 1
    fi
    break
  fi
  sleep 180
done

echo "[$(date -Is)] group comparison…"
B2B_EPOCH_ANCHOR="$ANCHOR" B2B_OUTDIR="$OUT" bash scripts/run_mne_ica_v1_largev3_group_compare.sh
echo "[$(date -Is)] DONE"
