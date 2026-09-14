#!/bin/bash
# Phase 3b/3c submitter.
#   bash scripts/submit_band_b2b.sh <delta|theta> <onset|offset> [all|left_temporal|right_temporal]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

BAND="${1:?usage: submit_band_b2b.sh band frame [chan_set]}"
FRAME="${2:?}"
CHAN="${3:-all}"
[[ "$BAND" == delta || "$BAND" == theta ]] || { echo "band must be delta|theta"; exit 1; }
[[ "$FRAME" == onset || "$FRAME" == offset ]] || { echo "frame must be onset|offset"; exit 1; }
[[ "$CHAN" == all || "$CHAN" == left_temporal || "$CHAN" == right_temporal ]] || {
  echo "chan must be all|left_temporal|right_temporal"; exit 1
}

# Drop leftover Phase-1 B2B_* so ALL-export cannot revive a passthrough outdir.
unset B2B_OUTDIR B2B_OBSERVED_OUTDIR B2B_EEG_BAND B2B_CHAN_SET B2B_EPOCH_ANCHOR

POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
export B2B_EXTRACTOR_DIR="${POOL}/extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b"
export B2B_EEG_BAND="$BAND"
export B2B_CHAN_SET="$CHAN"
export B2B_PASSTHROUGH=1
export B2B_H_RIDGE_KAPPA=0
export B2B_PCA_K=1
export B2B_PCA_FIT_MAX=1
export B2B_DRIFT_KNOTS=0
export B2B_INCLUDE_NUISANCE=0
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_STRICT_FEAT_DIMS=0
export B2B_NO_PLOT=1
export B2B_ALLOW_INPLACE_NULLS=1
export B2B_EXISTENCE_MIN_SHIFT=20
export B2B_EXISTENCE_N_NULL=200
export B2B_N_NULL=200
export B2B_FEATURE_SETS=envelope_v2:pitch:tone_v3:offset:lexical_frequency:gpt2cn_surprisal
export B2B_FAMILY_NAMES=envelope:pitch:tone:offset:frequency:surprisal

if [[ "$FRAME" == onset ]]; then
  export B2B_EPOCH_ANCHOR=onset
  export B2B_TMIN_S=-0.3
  export B2B_TMAX_S=1.0
  FROZEN="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_tonev3_onset_tmin03_passthrough"
  TAG="onset_tmin03"
else
  export B2B_EPOCH_ANCHOR=offset
  export B2B_TMIN_S=-0.5
  export B2B_TMAX_S=1.0
  FROZEN="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_tonev3_woffset_tmin05_passthrough"
  TAG="woffset_tmin05"
fi
export B2B_BASIS_DIR="${FROZEN}/_basis"

CHAN_TAG=""
[[ "$CHAN" != all ]] && CHAN_TAG="_${CHAN}"
export B2B_OUTDIR="${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_tonev3_${TAG}_band_${BAND}${CHAN_TAG}"
export B2B_OBSERVED_OUTDIR="${B2B_OUTDIR}"

[[ "$B2B_OUTDIR" == *passthrough* ]] && { echo "outdir leaked passthrough"; exit 1; }
test -f "${B2B_BASIS_DIR}/feature_basis.npz" || { echo "Missing basis ${B2B_BASIS_DIR}"; exit 1; }
mkdir -p "${B2B_OUTDIR}"

JOB_NAME="b2b_${BAND:0:1}${FRAME:0:2}${CHAN_TAG:0:3}"
PARTITION="${PARTITION:-mit_normal}"

EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR}"
EXPORT+=",B2B_OUTDIR=${B2B_OUTDIR}"
EXPORT+=",B2B_OBSERVED_OUTDIR=${B2B_OBSERVED_OUTDIR}"
EXPORT+=",B2B_BASIS_DIR=${B2B_BASIS_DIR}"
EXPORT+=",B2B_EEG_BAND=${B2B_EEG_BAND}"
EXPORT+=",B2B_CHAN_SET=${B2B_CHAN_SET}"
EXPORT+=",B2B_ALLOW_INPLACE_NULLS=1"
EXPORT+=",B2B_PASSTHROUGH=1,B2B_H_RIDGE_KAPPA=0,B2B_PCA_K=1,B2B_PCA_FIT_MAX=1,B2B_DRIFT_KNOTS=0"
EXPORT+=",B2B_EPOCH_ANCHOR=${B2B_EPOCH_ANCHOR},B2B_TMIN_S=${B2B_TMIN_S},B2B_TMAX_S=${B2B_TMAX_S}"
EXPORT+=",B2B_FEATURE_SETS=${B2B_FEATURE_SETS},B2B_FAMILY_NAMES=${B2B_FAMILY_NAMES}"
EXPORT+=",B2B_ALLOW_ZERO_FEATURE_ROWS=1,B2B_STRICT_FEAT_DIMS=0,B2B_NO_PLOT=1"
EXPORT+=",B2B_EXISTENCE_MIN_SHIFT=20,B2B_EXISTENCE_N_NULL=200,B2B_N_NULL=200"
EXPORT+=",JULIA_CPU_TARGET=generic"

echo "=== band B2B ${BAND} ${FRAME} ${CHAN} → ${B2B_OUTDIR} ==="
JOB_ID=$(sbatch --parsable --job-name="${JOB_NAME}" --partition="${PARTITION}" \
  --exclude=node1617 --export="${EXPORT}" scripts/run_band_b2b_array.sh)
echo "Submitted ${JOB_ID}"
echo "${JOB_ID}" >> logs/band_b2b_job_ids.txt
echo "${BAND} ${FRAME} ${CHAN} ${JOB_ID} ${B2B_OUTDIR}" >> logs/band_b2b_job_ids.txt
echo "${JOB_ID}"
