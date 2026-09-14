#!/bin/bash
# Launch offset (Large V2) and Tiny (onset) speech+language acoures B2B arms.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
chmod +x scripts/run_whisper_speechlang_acoures_offset_*.sh \
          scripts/run_whisper_speechlang_acoures_tiny_*.sh

J_OFF_BASIS=$(sbatch --parsable scripts/run_whisper_speechlang_acoures_offset_prepare_basis.sh)
J_OFF_OBS=$(sbatch --parsable --dependency=afterok:${J_OFF_BASIS} \
  scripts/run_whisper_speechlang_acoures_offset_observed_array.sh)

J_TINY_BASIS=$(sbatch --parsable scripts/run_whisper_speechlang_acoures_tiny_prepare_basis.sh)
J_TINY_OBS=$(sbatch --parsable --dependency=afterok:${J_TINY_BASIS} \
  scripts/run_whisper_speechlang_acoures_tiny_observed_array.sh)

echo "offset basis=${J_OFF_BASIS}  observed=${J_OFF_OBS}"
echo "tiny   basis=${J_TINY_BASIS}  observed=${J_TINY_OBS}"
echo "outdirs:"
echo "  encoding_results_b2b_largev2_offset_speechlang_acoures_k8"
echo "  encoding_results_b2b_whispertiny_onset_speechlang_acoures_k8"
