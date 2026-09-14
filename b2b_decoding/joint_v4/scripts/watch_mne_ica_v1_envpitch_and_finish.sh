#!/usr/bin/env bash
# Watch ICA env/pitch observed array, then submit nulls and group-compare.
#
#   OBS_JOB=<id> nohup bash scripts/watch_mne_ica_v1_envpitch_and_finish.sh \
#     > logs/watch_mne_ica_v1_envpitch.out 2>&1 &
#   B2B_EPOCH_ANCHOR=offset OBS_JOB=<id> nohup bash scripts/watch_mne_ica_v1_envpitch_and_finish.sh \
#     > logs/watch_mne_ica_v1_envpitch_offset.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
ANCHOR="${B2B_EPOCH_ANCHOR:-onset}"
if [[ -n "${B2B_OUTDIR:-}" ]]; then
  OUT="$B2B_OUTDIR"
elif [[ "$ANCHOR" == "offset" ]]; then
  OUT="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_offset_freq_surp_passthrough"
else
  OUT="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_onset_freq_surp_passthrough"
fi
if [[ "$ANCHOR" == "offset" ]]; then
  NULL_NAME="${NULL_NAME:-b2b_ica_envfn}"
else
  NULL_NAME="${NULL_NAME:-b2b_ica_envn}"
fi
OBS_JOB="${OBS_JOB:-}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b}"
export B2B_OUTDIR="$OUT"
export B2B_BASIS_DIR="${OUT}/_basis"
export B2B_PASSTHROUGH=1
export B2B_H_RIDGE_KAPPA=0
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_FEATURE_SETS="${B2B_FEATURE_SETS:-envelope_v2:pitch:word_onset:lexical_frequency:gpt2cn_surprisal}"
export B2B_FAMILY_NAMES="${B2B_FAMILY_NAMES:-envelope:pitch:onset:frequency:surprisal}"
export B2B_TMIN_S="${B2B_TMIN_S:--0.2}"
export B2B_TMAX_S="${B2B_TMAX_S:-1.0}"
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_STRICT_FEAT_DIMS=0
export B2B_NO_PLOT=1
export B2B_EPOCH_ANCHOR="$ANCHOR"
export B2B_BASIS_MODE=independent
export B2B_N_NULL=200
export B2B_NULL_COUNT=40

echo "[$(date -Is)] watching $OUT obs_job=${OBS_JOB:-none} anchor=$ANCHOR"
while true; do
  n=$(find "$OUT" -mindepth 2 -maxdepth 2 -name '*_b2b_family_agg.csv' 2>/dev/null | wc -l)
  echo "[$(date -Is)] n_agg=$n"
  [[ "$n" -ge 63 ]] && break
  sleep 180
done

EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR}"
EXPORT+=",B2B_OUTDIR=${OUT}"
EXPORT+=",B2B_BASIS_DIR=${B2B_BASIS_DIR}"
EXPORT+=",B2B_MODE=null"
EXPORT+=",B2B_PASSTHROUGH=1"
EXPORT+=",B2B_H_RIDGE_KAPPA=0"
EXPORT+=",B2B_PCA_K=1"
EXPORT+=",B2B_PCA_FIT_MAX=1"
EXPORT+=",B2B_DRIFT_KNOTS=0"
EXPORT+=",B2B_FEATURE_SETS=${B2B_FEATURE_SETS}"
EXPORT+=",B2B_FAMILY_NAMES=${B2B_FAMILY_NAMES}"
EXPORT+=",B2B_ALLOW_ZERO_FEATURE_ROWS=1"
EXPORT+=",B2B_STRICT_FEAT_DIMS=0"
EXPORT+=",B2B_NO_PLOT=1"
EXPORT+=",B2B_EPOCH_ANCHOR=${ANCHOR}"
EXPORT+=",B2B_TMIN_S=${B2B_TMIN_S}"
EXPORT+=",B2B_TMAX_S=${B2B_TMAX_S}"
EXPORT+=",B2B_N_NULL=200"
EXPORT+=",B2B_NULL_COUNT=40"

if [[ "${SKIP_NULL_SUBMIT:-0}" == "1" ]]; then
  NULL_JOB="${NULL_JOB:-}"
  echo "[$(date -Is)] skip null submit; NULL_JOB=${NULL_JOB:-none}"
else
  echo "[$(date -Is)] observed complete. Submitting nulls…"
  NULL_JOB=$(sbatch --parsable --job-name="${NULL_NAME}" \
    --time="${NULL_TIME:-4:00:00}" --export="${EXPORT}" \
    scripts/run_null_array.sh)
  echo "NULL_JOB=$NULL_JOB"
  echo "$NULL_JOB" > "logs/mne_ica_v1_envpitch_${ANCHOR}_null_job_id.txt"
  if [[ "$ANCHOR" == "onset" ]]; then
    echo "$NULL_JOB" > logs/mne_ica_v1_envpitch_null_job_id.txt
  fi
fi
[[ -n "${NULL_JOB}" ]] || { echo "ERROR: no NULL_JOB"; exit 1; }

while true; do
  nnull=$(find "$OUT" -mindepth 2 -maxdepth 2 -name '*_null_traces.csv' 2>/dev/null | wc -l)
  echo "[$(date -Is)] n_null_agg=$nnull"
  [[ "$nnull" -ge 63 ]] && break
  left=$(squeue -j "$NULL_JOB" -h 2>/dev/null | wc -l || true)
  echo "  null tasks left≈$left"
  if [[ "$left" -eq 0 && "$nnull" -lt 63 ]]; then
    module load julia/1.12.6
    B2B_OUTDIR="$OUT" julia --project=.. aggregate_v4.jl --nulls-only || true
    nnull=$(find "$OUT" -mindepth 2 -maxdepth 2 -name '*_null_traces.csv' 2>/dev/null | wc -l)
    [[ "$nnull" -ge 63 ]] && break
    echo "ERROR: nulls incomplete ($nnull/63)" >&2
    exit 1
  fi
  sleep 180
done

B2B_EPOCH_ANCHOR="$ANCHOR" B2B_OUTDIR="$OUT" bash scripts/run_mne_ica_v1_envpitch_group_compare.sh
echo "[$(date -Is)] DONE"
