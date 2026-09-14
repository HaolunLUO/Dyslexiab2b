#!/bin/bash
#SBATCH -J b2b_v3_pilot
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o logs/b2b_largev3_pilot_%j.out
#SBATCH -e logs/b2b_largev3_pilot_%j.err
#
# Wire large-v3 features and run 3-subject B2B pilot (RN109, D007d, D011d).
# Submit from joint_v4/: mkdir -p logs && sbatch scripts/submit_whisperlargev3_pilot.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p logs

POOL="${POOL:-/orcd/pool/005/haolun52}"
export POOL

bash scripts/wire_whisperlargev3_b2b_extractors.sh
bash scripts/run_whisperlargev3_pilot.sh
