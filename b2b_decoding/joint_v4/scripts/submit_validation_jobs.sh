#!/usr/bin/env bash
# Submit validation Slurm jobs when the controller is back.
#   bash scripts/submit_validation_jobs.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

scontrol ping | head -2

echo "Submitting HDC dss40_bin20 null array (315 tasks)…"
NULL_JOB=$(sbatch --export=NONE --parsable scripts/run_hdc_snr_dss_bin_null_array.sh)
echo "null job: $NULL_JOB"

echo "Submitting DSS train_half observed array (63 tasks)…"
TH_JOB=$(sbatch --export=NONE --parsable scripts/run_hdc_snr_dss_trainhalf_observed_array.sh)
echo "train_half job: $TH_JOB"

cat > validation_cap_dss40_bin20/slurm_jobs.json <<EOF
{"null_array": "$NULL_JOB", "train_half_observed": "$TH_JOB", "submitted_at": "$(date -Is)"}
EOF
echo "Wrote validation_cap_dss40_bin20/slurm_jobs.json"
squeue -u "$USER" | head -20
