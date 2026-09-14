#!/bin/bash
# Submit remaining Phase 3b/3c arms after delta-onset-all is healthy.
# Does not submit delta onset all (that is the first arm).
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/submit_band_b2b.sh delta offset all
bash scripts/submit_band_b2b.sh theta onset all
bash scripts/submit_band_b2b.sh theta offset all
bash scripts/submit_band_b2b.sh delta onset left_temporal
bash scripts/submit_band_b2b.sh delta onset right_temporal
echo "Submitted remaining Phase 3b/3c arrays (see logs/band_b2b_job_ids.txt)"
