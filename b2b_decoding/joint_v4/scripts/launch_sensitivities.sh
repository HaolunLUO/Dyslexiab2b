#!/bin/bash
# Launch all joint_v4 sensitivity arms: basis jobs, then observed arrays.
# Usage (from joint_v4/): bash scripts/launch_sensitivities.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
POOL=/home/haolun52/orcd/pool
EXTRACTOR="${B2B_EXTRACTOR_DIR:-$POOL/extracted_sections_wordlocked_shared}"

submit_basis() {
  local name="$1"
  shift
  # Remaining args are KEY=VAL exports for the job
  local export_list="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
  local kv
  for kv in "$@"; do
    export_list+=",${kv}"
  done
  sbatch --job-name="b2b_v4_basis_${name}" \
    --export="${export_list}" \
    scripts/run_prepare_basis.sh
}

echo "=== Submitting basis jobs ==="
# K=4 independent onset
J_K4=$(submit_basis k4 \
  "B2B_PCA_K=4" \
  "B2B_BASIS_MODE=independent" \
  "B2B_EPOCH_ANCHOR=onset" \
  "B2B_FEAT_LANGUAGE=whisperlargev2_language_audio_fused" \
  "B2B_OUTDIR=${POOL}/encoding_results_b2b_largev2_onset_joint_k4" \
  "B2B_BASIS_DIR=${POOL}/encoding_results_b2b_largev2_onset_joint_k4/_basis" \
  "B2B_EXTRACTOR_DIR=${EXTRACTOR}" | awk '{print $4}')
echo "  k4 basis → job ${J_K4}"

# K=16 independent onset
J_K16=$(submit_basis k16 \
  "B2B_PCA_K=16" \
  "B2B_BASIS_MODE=independent" \
  "B2B_EPOCH_ANCHOR=onset" \
  "B2B_FEAT_LANGUAGE=whisperlargev2_language_audio_fused" \
  "B2B_OUTDIR=${POOL}/encoding_results_b2b_largev2_onset_joint_k16" \
  "B2B_BASIS_DIR=${POOL}/encoding_results_b2b_largev2_onset_joint_k16/_basis" \
  "B2B_EXTRACTOR_DIR=${EXTRACTOR}" | awk '{print $4}')
echo "  k16 basis → job ${J_K16}"

# Text-only language
J_TEXT=$(submit_basis text \
  "B2B_PCA_K=8" \
  "B2B_BASIS_MODE=independent" \
  "B2B_EPOCH_ANCHOR=onset" \
  "B2B_FEAT_LANGUAGE=whisperlargev2_language_text_only" \
  "B2B_OUTDIR=${POOL}/encoding_results_b2b_largev2_onset_textonly_k8" \
  "B2B_BASIS_DIR=${POOL}/encoding_results_b2b_largev2_onset_textonly_k8/_basis" \
  "B2B_EXTRACTOR_DIR=${EXTRACTOR}" | awk '{print $4}')
echo "  textonly basis → job ${J_TEXT}"

# Ordered residualization
J_ORD=$(submit_basis ord \
  "B2B_PCA_K=8" \
  "B2B_BASIS_MODE=ordered" \
  "B2B_EPOCH_ANCHOR=onset" \
  "B2B_FEAT_LANGUAGE=whisperlargev2_language_audio_fused" \
  "B2B_OUTDIR=${POOL}/encoding_results_b2b_largev2_onset_ordered_k8" \
  "B2B_BASIS_DIR=${POOL}/encoding_results_b2b_largev2_onset_ordered_k8/_basis" \
  "B2B_EXTRACTOR_DIR=${EXTRACTOR}" | awk '{print $4}')
echo "  ordered basis → job ${J_ORD}"

# Offset-locked (basis math same as primary; separate outdir/manifest)
J_OFF=$(submit_basis offset \
  "B2B_PCA_K=8" \
  "B2B_BASIS_MODE=independent" \
  "B2B_EPOCH_ANCHOR=offset" \
  "B2B_FEAT_LANGUAGE=whisperlargev2_language_audio_fused" \
  "B2B_OUTDIR=${POOL}/encoding_results_b2b_largev2_offset_joint_k8" \
  "B2B_BASIS_DIR=${POOL}/encoding_results_b2b_largev2_offset_joint_k8/_basis" \
  "B2B_EXTRACTOR_DIR=${EXTRACTOR}" | awk '{print $4}')
echo "  offset basis → job ${J_OFF}"

echo "=== Submitting observed arrays (afterok bases) ==="
# Bundled K4 / K16 / textonly / ordered (252 tasks)
J_SENS=$(sbatch --dependency="afterok:${J_K4}:${J_K16}:${J_TEXT}:${J_ORD}" \
  --export="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH},B2B_EXTRACTOR_DIR=${EXTRACTOR}" \
  scripts/run_sensitivity_array.sh | awk '{print $4}')
echo "  sensitivity observed array → job ${J_SENS} (depends on ${J_K4},${J_K16},${J_TEXT},${J_ORD})"

# Offset observed (63 tasks)
J_OFF_OBS=$(sbatch --dependency="afterok:${J_OFF}" \
  --job-name="b2b_v4_obs_offset" \
  --export="ALL,JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH},B2B_MODE=observed,B2B_PCA_K=8,B2B_BASIS_MODE=independent,B2B_EPOCH_ANCHOR=offset,B2B_FEAT_ACOUSTIC=whisperlargev2_acoustic,B2B_FEAT_SPEECH=whisperlargev2_speech,B2B_FEAT_LANGUAGE=whisperlargev2_language_audio_fused,B2B_OUTDIR=${POOL}/encoding_results_b2b_largev2_offset_joint_k8,B2B_BASIS_DIR=${POOL}/encoding_results_b2b_largev2_offset_joint_k8/_basis,B2B_EXTRACTOR_DIR=${EXTRACTOR},B2B_NO_PLOT=1" \
  scripts/run_observed_array.sh | awk '{print $4}')
echo "  offset observed array → job ${J_OFF_OBS} (depends on ${J_OFF})"

cat > logs/sensitivity_launch_ids.txt <<EOF
basis_k4=${J_K4}
basis_k16=${J_K16}
basis_text=${J_TEXT}
basis_ordered=${J_ORD}
basis_offset=${J_OFF}
observed_sensitivity_array=${J_SENS}
observed_offset_array=${J_OFF_OBS}
EOF

echo
echo "Wrote logs/sensitivity_launch_ids.txt"
echo "Monitor: squeue -u \$USER"
echo "After all observed finish, run group inference per outdir (no nulls for sensitivities)."
