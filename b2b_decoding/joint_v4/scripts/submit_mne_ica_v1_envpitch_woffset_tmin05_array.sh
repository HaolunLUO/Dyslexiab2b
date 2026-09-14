#!/bin/bash
# ICA v1 B2B: envelope / pitch / word-offset / frequency / surprisal.
# Offset-locked epochs with a wider window (−0.5 … 1.0 s).
#
# Usage (from joint_v4/):
#   mkdir -p logs
#   bash scripts/submit_mne_ica_v1_envpitch_woffset_tmin05_array.sh
#
# Optional: PARTITION=mit_preemptable
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${POOL}/.julia}"
PY="${PY:-$(pwd)/.venv_gpt2/bin/python}"

export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-${POOL}/extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b}"
export B2B_EPOCH_ANCHOR=offset
export B2B_TMIN_S=-0.5
export B2B_TMAX_S=1.0
export B2B_OUTDIR="${B2B_OUTDIR:-${POOL}/encoding_results_b2b_mne_ica_v1_envpitch_woffset_tmin05_passthrough}"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${B2B_OUTDIR}/_basis}"
export B2B_MODE=observed
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
export B2B_FEATURE_SETS=envelope_v2:pitch:offset:lexical_frequency:gpt2cn_surprisal
export B2B_FAMILY_NAMES=envelope:pitch:offset:frequency:surprisal
JOB_NAME="${JOB_NAME:-b2b_ica_woff}"
PARTITION="${PARTITION:-mit_normal}"

SHARED="${POOL}/extracted_sections_wordlocked_shared/_shared_wordlocked_features"
if [[ ! -f "${SHARED}/section_001/X_word_offset.npy" ]]; then
  echo "Extracting pitch + word-onset/offset features…"
  "$PY" scripts/extract_pitch_onset_wordlocked.py
fi

bash scripts/wire_mne_ica_v1_envpitch_b2b_extractors.sh

mkdir -p "${B2B_BASIS_DIR}"
if [[ ! -f "${B2B_BASIS_DIR}/feature_basis.npz" ]]; then
  echo "Preparing passthrough feature basis (offset feat, tmin=-0.5)…"
  module load julia/1.12.6
  julia --project=.. prepare_feature_basis.jl 2>&1 | tee logs/mne_ica_v1_envpitch_woffset_tmin05_prepare_basis.out
else
  echo "Basis already present: ${B2B_BASIS_DIR}/feature_basis.npz"
fi

EXPORT="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
EXPORT+=",B2B_EXTRACTOR_DIR=${B2B_EXTRACTOR_DIR}"
EXPORT+=",B2B_OUTDIR=${B2B_OUTDIR}"
EXPORT+=",B2B_BASIS_DIR=${B2B_BASIS_DIR}"
EXPORT+=",B2B_MODE=${B2B_MODE}"
EXPORT+=",B2B_PASSTHROUGH=1"
EXPORT+=",B2B_H_RIDGE_KAPPA=0"
EXPORT+=",B2B_PCA_K=1"
EXPORT+=",B2B_PCA_FIT_MAX=1"
EXPORT+=",B2B_DRIFT_KNOTS=0"
EXPORT+=",B2B_EPOCH_ANCHOR=${B2B_EPOCH_ANCHOR}"
EXPORT+=",B2B_TMIN_S=${B2B_TMIN_S}"
EXPORT+=",B2B_TMAX_S=${B2B_TMAX_S}"
EXPORT+=",B2B_FEATURE_SETS=${B2B_FEATURE_SETS}"
EXPORT+=",B2B_FAMILY_NAMES=${B2B_FAMILY_NAMES}"
EXPORT+=",B2B_ALLOW_ZERO_FEATURE_ROWS=1"
EXPORT+=",B2B_STRICT_FEAT_DIMS=0"
EXPORT+=",B2B_NO_PLOT=1"
EXPORT+=",JULIA_CPU_TARGET=generic"

echo "=== ICA env/pitch B2B (offset, word_offset feat, ${B2B_TMIN_S}…${B2B_TMAX_S} s) ==="
echo "  extractor: ${B2B_EXTRACTOR_DIR}"
echo "  outdir   : ${B2B_OUTDIR}"
echo "  partition: ${PARTITION}"
echo "  features : ${B2B_FEATURE_SETS}"

JOB_ID=$(sbatch --parsable --job-name="${JOB_NAME}" --partition="${PARTITION}" \
  --export="${EXPORT}" \
  scripts/run_mne_ica_v1_envpitch_observed_array.sh)
echo "Submitted ICA env/pitch wide-offset observed array → job ${JOB_ID}"
echo "${JOB_ID}" > logs/mne_ica_v1_envpitch_woffset_tmin05_array_job_id.txt
echo "${JOB_ID}"
