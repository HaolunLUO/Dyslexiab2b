#!/bin/bash
#SBATCH -J mfa_b2b_pilot
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/mfa_phone_tone_env_pilot_%j.out
#SBATCH -e logs/mfa_phone_tone_env_pilot_%j.err
#
# Onset-locked B2B pilot: envelope + MFA phones + tone (passthrough).
#   mkdir -p logs && sbatch scripts/submit_mfa_phone_tone_env_pilot.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export B2B_BLAS_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash scripts/run_mfa_phone_tone_env_pilot.sh "$@"
