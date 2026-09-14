#!/usr/bin/env bash
#SBATCH -J mfa_lppcn
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/extract_mfa_phone_tone_env_%j.out
#SBATCH -e logs/extract_mfa_phone_tone_env_%j.err
#
# MFA Mandarin phones + tone + Hilbert envelope, locked to existing
# word_timing_relative.csv onsets (1753 + 1800).
#
#   mkdir -p logs && sbatch scripts/submit_extract_mfa_phone_tone_envelope.sh

set -euo pipefail

POOL="${POOL:-/orcd/pool/005/haolun52}"
ROOT_DIR="${ROOT_DIR:-$POOL/dyslexia_natualistics_listing/b2b_decoding/joint_v4}"
SHARED_DIR="${SHARED_DIR:-$POOL/extracted_sections_wordlocked_shared/_shared_wordlocked_features}"
WORK_DIR="${WORK_DIR:-$POOL/mfa_lppcn_wordlocked}"
MFA_ROOT="${MFA_ROOT:-$POOL/mfa_root}"
SIF="${SIF:-$WORK_DIR/montreal-forced-aligner.sif}"
VENV_PY="${VENV_PY:-$ROOT_DIR/.venv_gpt2/bin/python}"
NUM_JOBS="${NUM_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

# If this script is copied to /var/spool/slurmd, stay in the repo.
if [[ -d "$ROOT_DIR" ]]; then
  cd "$ROOT_DIR"
fi
mkdir -p logs "$WORK_DIR" "$MFA_ROOT"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$WORK_DIR/.apptainer}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}"
mkdir -p "$APPTAINER_CACHEDIR"

echo "POOL=$POOL"
echo "SHARED_DIR=$SHARED_DIR"
echo "SIF=$SIF"
echo "MFA_ROOT=$MFA_ROOT"
echo "host=$(hostname)  cpus=$NUM_JOBS"

if command -v module >/dev/null 2>&1; then
  module load apptainer/1.5.2 >/dev/null 2>&1 || module load apptainer >/dev/null 2>&1 || true
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found after module load"
  command -v singularity && echo "(singularity is present; not using it)"
  exit 1
fi
echo "apptainer=$(command -v apptainer)"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: missing python $VENV_PY"
  exit 1
fi

MFA_IMAGE="${MFA_IMAGE:-docker://mmcauliffe/montreal-forced-aligner:v3.4.2}"

if [[ ! -f "$SIF" ]]; then
  echo "Pulling MFA apptainer image (first run): $MFA_IMAGE"
  apptainer pull "$SIF" "$MFA_IMAGE"
fi

SKIP_ALIGN_FLAG=()
if [[ "${SKIP_ALIGN:-1}" == "1" && -f "$WORK_DIR/aligned/section_001.TextGrid" ]]; then
  SKIP_ALIGN_FLAG=(--skip-align)
  echo "Reusing existing TextGrids under $WORK_DIR/aligned"
fi

set -x
"$VENV_PY" "$ROOT_DIR/scripts/extract_mfa_phone_tone_envelope_wordlocked.py" \
  --aligner mfa \
  --shared-dir "$SHARED_DIR" \
  --sections 1 2 \
  --work-dir "$WORK_DIR" \
  --mfa-sif "$SIF" \
  --mfa-root "$MFA_ROOT" \
  --textgrid-dir "$WORK_DIR/aligned" \
  --feature-tag v2 \
  --num-jobs "$NUM_JOBS" \
  --beam 100 \
  "${SKIP_ALIGN_FLAG[@]}"
set +x

echo "MFA phone/tone/envelope v2 extraction complete."
ls -l "$SHARED_DIR"/section_001/X_word_{envelope_v2,mfa_phones_v2,mfa_tone_v2}.npy
# Leave legacy v1 keys untouched; mark audit note.
LEGACY_NOTE="$SHARED_DIR/SUPERSEDED_mfa_v1_features.txt"
if [[ ! -f "$LEGACY_NOTE" ]]; then
  cat > "$LEGACY_NOTE" <<EOF
Legacy MFA feature keys (envelope / mfa_phones / mfa_tone) are SUPERSEDED.
Use envelope_v2 / mfa_phones_v2 / mfa_tone_v2 for analysis.
Reason: broken Chao parsing, unclipped phone durations, env_onset==env_t00, rank-deficient design.
Preserved for auditability; do not overwrite.
EOF
fi
"$VENV_PY" "$ROOT_DIR/scripts/preflight_mfa_v2_features.py" \
  --shared-dir "$SHARED_DIR" \
  --keys envelope_v2,mfa_phones_v2,mfa_tone_v2 \
  --out-json "$SHARED_DIR/mfa_v2_preflight_qc.json"
